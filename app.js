// Main application logic for Two-Tower Recommender

class MovieLensApp {
    constructor() {
        this.data = null;
        this.baselineModel = null;
        this.deepModel = null;
        this.isTraining = false;
        
        // Data structures
        this.userMap = new Map(); // user_id -> index
        this.itemMap = new Map(); // item_id -> index
        this.userItems = new Map(); // user_idx -> Set of item_idx
        this.itemGenres = []; // item_idx -> genre vector
        this.itemTitles = []; // item_idx -> title
        
        // Reverse mappings
        this.reverseUserMap = [];
        this.reverseItemMap = [];
        
        // Training state
        this.lossHistory = { baseline: [], deep: [] };
        this.currentEpoch = 0;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
    }

    async loadData() {
        this.updateStatus('Loading MovieLens data...');
        
        try {
            // Load item data
            const itemResponse = await fetch('./data/u.item');
            const itemText = await itemResponse.text();
            await this.parseItemData(itemText);
            
            // Load interaction data
            const dataResponse = await fetch('./data/u.data');
            const dataText = await dataResponse.text();
            await this.parseInteractionData(dataText);
            
            this.updateStatus(`Data loaded: ${this.userMap.size} users, ${this.itemMap.size} items, ${this.data.length} interactions`);
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
            console.error(error);
        }
    }

    parseItemData(itemText) {
        const lines = itemText.split('\n').filter(line => line.trim());
        
        this.itemGenres = [];
        this.itemTitles = [];
        this.reverseItemMap = [];
        
        lines.forEach(line => {
            const parts = line.split('|');
            if (parts.length >= 24) {
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const genres = parts.slice(5, 24).map(x => parseInt(x)); // 19 genre flags
                
                const itemIdx = this.itemMap.size;
                this.itemMap.set(itemId, itemIdx);
                this.reverseItemMap[itemIdx] = itemId;
                this.itemTitles[itemIdx] = title;
                this.itemGenres[itemIdx] = genres;
            }
        });
    }

    parseInteractionData(dataText) {
        const lines = dataText.split('\n').filter(line => line.trim());
        const maxInteractions = parseInt(document.getElementById('maxInteractions').value);
        
        this.data = [];
        this.userMap.clear();
        this.userItems.clear();
        this.reverseUserMap = [];
        
        // First pass: collect all users and count interactions
        const userInteractions = new Map();
        
        lines.slice(0, maxInteractions).forEach(line => {
            const [userId, itemId, rating, timestamp] = line.split('\t').map(x => parseInt(x));
            
            if (!userInteractions.has(userId)) {
                userInteractions.set(userId, []);
            }
            userInteractions.get(userId).push({ itemId, rating, timestamp });
        });
        
        // Create user mapping
        Array.from(userInteractions.keys()).forEach(userId => {
            const userIdx = this.userMap.size;
            this.userMap.set(userId, userIdx);
            this.reverseUserMap[userIdx] = userId;
            this.userItems.set(userIdx, new Set());
        });
        
        // Second pass: populate data and userItems
        lines.slice(0, maxInteractions).forEach(line => {
            const [userId, itemId, rating, timestamp] = line.split('\t').map(x => parseInt(x));
            
            if (this.userMap.has(userId) && this.itemMap.has(itemId)) {
                const userIdx = this.userMap.get(userId);
                const itemIdx = this.itemMap.get(itemId);
                
                this.data.push({
                    userIdx,
                    itemIdx,
                    rating,
                    timestamp
                });
                
                this.userItems.get(userIdx).add(itemIdx);
            }
        });
        
        console.log(`Parsed ${this.data.length} interactions`);
    }

    async train() {
        if (!this.data) {
            this.updateStatus('Please load data first');
            return;
        }
        
        if (this.isTraining) {
            this.updateStatus('Training already in progress');
            return;
        }
        
        this.isTraining = true;
        this.updateStatus('Starting training...');
        
        const params = this.getTrainingParams();
        const trainBaseline = document.getElementById('trainBaseline').value === 'yes';
        
        // Initialize models
        if (trainBaseline) {
            this.baselineModel = new TwoTowerBaseline(
                this.userMap.size,
                this.itemMap.size,
                params.embeddingDim
            );
        }
        
        this.deepModel = new TwoTowerDeep(
            this.userMap.size,
            this.itemMap.size,
            params.embeddingDim,
            params.hiddenDim,
            this.itemGenres[0].length
        );
        
        // Prepare training data
        const interactions = this.data.map(d => ({
            userIdx: d.userIdx,
            itemIdx: d.itemIdx
        }));
        
        // Training loop
        this.lossHistory = { baseline: [], deep: [] };
        this.currentEpoch = 0;
        
        for (let epoch = 0; epoch < params.epochs; epoch++) {
            this.currentEpoch = epoch;
            await this.trainEpoch(interactions, params, trainBaseline);
            
            if (!this.isTraining) break; // Allow cancellation
        }
        
        this.isTraining = false;
        this.updateStatus('Training completed');
        
        // Draw PCA after training
        this.drawItemEmbeddingsPCA();
    }

    async trainEpoch(interactions, params, trainBaseline) {
        const batchSize = params.batchSize;
        const numBatches = Math.ceil(interactions.length / batchSize);
        
        // Shuffle interactions
        const shuffled = [...interactions].sort(() => Math.random() - 0.5);
        
        for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            const start = batchIdx * batchSize;
            const end = Math.min(start + batchSize, interactions.length);
            const batch = shuffled.slice(start, end);
            
            if (params.lossType === 'softmax') {
                await this.trainBatchSoftmax(batch, params, trainBaseline);
            } else {
                await this.trainBatchBPR(batch, params, trainBaseline);
            }
            
            // Update loss chart every few batches
            if (batchIdx % 10 === 0) {
                this.drawLossChart();
            }
            
            // Yield to UI
            await new Promise(resolve => setTimeout(resolve, 0));
            
            if (!this.isTraining) break;
        }
    }

    async trainBatchSoftmax(batch, params, trainBaseline) {
        const userIndices = batch.map(b => b.userIdx);
        const itemIndices = batch.map(b => b.itemIdx);
        
        const userTensor = tf.tensor2d(userIndices, [userIndices.length, 1], 'int32');
        const itemTensor = tf.tensor2d(itemIndices, [itemIndices.length, 1], 'int32');
        
        const optimizer = TrainingUtils.createOptimizer(params.learningRate);
        
        // Train baseline model
        if (trainBaseline && this.baselineModel) {
            const baselineLoss = optimizer.minimize(() => {
                const userEmbs = this.baselineModel.userForward(userTensor);
                const itemEmbs = this.baselineModel.itemForward(itemTensor);
                return LossFunctions.inBatchSoftmaxLoss(userEmbs, itemEmbs);
            }, true);
            
            if (baselineLoss) {
                this.lossHistory.baseline.push({
                    epoch: this.currentEpoch,
                    batch: this.lossHistory.baseline.length,
                    loss: baselineLoss.dataSync()[0]
                });
                baselineLoss.dispose();
            }
        }
        
        // Train deep model
        const genreTensor = tf.tensor2d(
            itemIndices.map(idx => this.itemGenres[idx]),
            [itemIndices.length, this.itemGenres[0].length]
        );
        
        const deepLoss = optimizer.minimize(() => {
            const userEmbs = this.deepModel.userForward(userTensor);
            const itemEmbs = this.deepModel.itemForward(itemTensor, genreTensor);
            return LossFunctions.inBatchSoftmaxLoss(userEmbs, itemEmbs);
        }, true);
        
        if (deepLoss) {
            this.lossHistory.deep.push({
                epoch: this.currentEpoch,
                batch: this.lossHistory.deep.length,
                loss: deepLoss.dataSync()[0]
            });
            deepLoss.dispose();
        }
        
        // Cleanup
        tf.dispose([userTensor, itemTensor, genreTensor]);
    }

    async trainBatchBPR(batch, params, trainBaseline) {
        // For BPR, we need negative samples
        const userIndices = [];
        const posItemIndices = [];
        const negItemIndices = [];
        
        for (const interaction of batch) {
            const userIdx = interaction.userIdx;
            const posItemIdx = interaction.itemIdx;
            
            // Sample negative item
            const ratedItems = this.userItems.get(userIdx);
            let negItemIdx;
            do {
                negItemIdx = Math.floor(Math.random() * this.itemMap.size);
            } while (ratedItems.has(negItemIdx));
            
            userIndices.push(userIdx);
            posItemIndices.push(posItemIdx);
            negItemIndices.push(negItemIdx);
        }
        
        const userTensor = tf.tensor2d(userIndices, [userIndices.length, 1], 'int32');
        const posItemTensor = tf.tensor2d(posItemIndices, [posItemIndices.length, 1], 'int32');
        const negItemTensor = tf.tensor2d(negItemIndices, [negItemIndices.length, 1], 'int32');
        
        const optimizer = TrainingUtils.createOptimizer(params.learningRate);
        
        // Train baseline model
        if (trainBaseline && this.baselineModel) {
            const baselineLoss = optimizer.minimize(() => {
                const userEmbs = this.baselineModel.userForward(userTensor);
                const posEmbs = this.baselineModel.itemForward(posItemTensor);
                const negEmbs = this.baselineModel.itemForward(negItemTensor);
                return LossFunctions.bprLoss(userEmbs, posEmbs, negEmbs);
            }, true);
            
            if (baselineLoss) {
                this.lossHistory.baseline.push({
                    epoch: this.currentEpoch,
                    batch: this.lossHistory.baseline.length,
                    loss: baselineLoss.dataSync()[0]
                });
                baselineLoss.dispose();
            }
        }
        
        // Train deep model
        const posGenreTensor = tf.tensor2d(
            posItemIndices.map(idx => this.itemGenres[idx]),
            [posItemIndices.length, this.itemGenres[0].length]
        );
        const negGenreTensor = tf.tensor2d(
            negItemIndices.map(idx => this.itemGenres[idx]),
            [negItemIndices.length, this.itemGenres[0].length]
        );
        
        const deepLoss = optimizer.minimize(() => {
            const userEmbs = this.deepModel.userForward(userTensor);
            const posEmbs = this.deepModel.itemForward(posItemTensor, posGenreTensor);
            const negEmbs = this.deepModel.itemForward(negItemTensor, negGenreTensor);
            return LossFunctions.bprLoss(userEmbs, posEmbs, negEmbs);
        }, true);
        
        if (deepLoss) {
            this.lossHistory.deep.push({
                epoch: this.currentEpoch,
                batch: this.lossHistory.deep.length,
                loss: deepLoss.dataSync()[0]
            });
            deepLoss.dispose();
        }
        
        // Cleanup
        tf.dispose([userTensor, posItemTensor, negItemTensor, posGenreTensor, negGenreTensor]);
    }

    async test() {
        if (!this.deepModel) {
            this.updateStatus('Please train models first');
            return;
        }
        
        this.updateStatus('Generating recommendations...');
        
        // Find a user with at least 20 ratings
        const eligibleUsers = Array.from(this.userItems.entries())
            .filter(([userIdx, items]) => items.size >= 20)
            .map(([userIdx]) => userIdx);
        
        if (eligibleUsers.length === 0) {
            this.updateStatus('No users with sufficient ratings found');
            return;
        }
        
        const testUserIdx = eligibleUsers[Math.floor(Math.random() * eligibleUsers.length)];
        const ratedItems = Array.from(this.userItems.get(testUserIdx));
        
        // Get user's top rated items (historical)
        const userRatings = this.data
            .filter(d => d.userIdx === testUserIdx)
            .sort((a, b) => b.rating - a.rating || b.timestamp - a.timestamp)
            .slice(0, 10);
        
        // Generate recommendations
        const baselineRecs = this.baselineModel ? 
            await this.generateRecommendations(testUserIdx, ratedItems, 'baseline') : [];
        const deepRecs = await this.generateRecommendations(testUserIdx, ratedItems, 'deep');
        
        // Render results
        this.renderComparisonTable(userRatings, baselineRecs, deepRecs);
        this.updateStatus(`Recommendations generated for user ${this.reverseUserMap[testUserIdx]}`);
    }

    async generateRecommendations(userIdx, ratedItems, modelType) {
        const model = modelType === 'baseline' ? this.baselineModel : this.deepModel;
        if (!model) return [];
        
        const userTensor = tf.tensor2d([userIdx], [1, 1], 'int32');
        const scores = [];
        
        // Score all unrated items
        for (let itemIdx = 0; itemIdx < this.itemMap.size; itemIdx++) {
            if (!ratedItems.includes(itemIdx)) {
                let score;
                if (modelType === 'baseline') {
                    const itemTensor = tf.tensor2d([itemIdx], [1, 1], 'int32');
                    score = await model.predict(userTensor, itemTensor).data();
                    itemTensor.dispose();
                } else {
                    const itemTensor = tf.tensor2d([itemIdx], [1, 1], 'int32');
                    const genreTensor = tf.tensor2d([this.itemGenres[itemIdx]], [1, this.itemGenres[0].length]);
                    score = await model.predict(userTensor, itemTensor, genreTensor).data();
                    itemTensor.dispose();
                    genreTensor.dispose();
                }
                scores.push({ itemIdx, score: score[0] });
            }
        }
        
        userTensor.dispose();
        
        // Return top 10
        return scores
            .sort((a, b) => b.score - a.score)
            .slice(0, 10)
            .map(rec => ({
                itemIdx: rec.itemIdx,
                score: rec.score,
                title: this.itemTitles[rec.itemIdx]
            }));
    }

    renderComparisonTable(historical, baselineRecs, deepRecs) {
        const container = document.getElementById('results');
        
        const html = `
            <div class="comparison-table">
                <div class="table-column">
                    <h4>Top-10 Rated (Historical)</h4>
                    <table>
                        <thead><tr><th>Movie</th><th>Rating</th></tr></thead>
                        <tbody>
                            ${historical.map(item => `
                                <tr>
                                    <td>${this.itemTitles[item.itemIdx]}</td>
                                    <td>${item.rating}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div class="table-column">
                    <h4>Top-10 Recommended (Baseline)</h4>
                    <table>
                        <thead><tr><th>Movie</th><th>Score</th></tr></thead>
                        <tbody>
                            ${baselineRecs.map(rec => `
                                <tr>
                                    <td>${rec.title}</td>
                                    <td>${rec.score.toFixed(4)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div class="table-column">
                    <h4>Top-10 Recommended (Deep)</h4>
                    <table>
                        <thead><tr><th>Movie</th><th>Score</th></tr></thead>
                        <tbody>
                            ${deepRecs.map(rec => `
                                <tr>
                                    <td>${rec.title}</td>
                                    <td>${rec.score.toFixed(4)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
    }

    drawLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        if (this.lossHistory.baseline.length === 0 && this.lossHistory.deep.length === 0) {
            return;
        }
        
        // Find data bounds
        const allLosses = [
            ...this.lossHistory.baseline.map(d => d.loss),
            ...this.lossHistory.deep.map(d => d.loss)
        ];
        const minLoss = Math.min(...allLosses);
        const maxLoss = Math.max(...allLosses);
        
        const padding = 50;
        const chartWidth = canvas.width - 2 * padding;
        const chartHeight = canvas.height - 2 * padding;
        
        // Draw axes
        ctx.strokeStyle = '#ccc';
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, canvas.height - padding);
        ctx.lineTo(canvas.width - padding, canvas.height - padding);
        ctx.stroke();
        
        // Draw baseline loss
        if (this.lossHistory.baseline.length > 0) {
            ctx.strokeStyle = 'green';
            ctx.beginPath();
            this.lossHistory.baseline.forEach((point, i) => {
                const x = padding + (i / this.lossHistory.baseline.length) * chartWidth;
                const y = canvas.height - padding - ((point.loss - minLoss) / (maxLoss - minLoss)) * chartHeight;
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        // Draw deep loss
        if (this.lossHistory.deep.length > 0) {
            ctx.strokeStyle = 'red';
            ctx.beginPath();
            this.lossHistory.deep.forEach((point, i) => {
                const x = padding + (i / this.lossHistory.deep.length) * chartWidth;
                const y = canvas.height - padding - ((point.loss - minLoss) / (maxLoss - minLoss)) * chartHeight;
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        // Draw legend
        ctx.fillStyle = 'black';
        ctx.fillText('Loss', 10, canvas.height / 2);
        ctx.fillText('Batch', canvas.width / 2, canvas.height - 10);
        
        if (this.lossHistory.baseline.length > 0) {
            ctx.fillStyle = 'green';
            ctx.fillText('Baseline', canvas.width - 100, 20);
        }
        
        if (this.lossHistory.deep.length > 0) {
            ctx.fillStyle = 'red';
            ctx.fillText('Deep', canvas.width - 100, 40);
        }
    }

    drawItemEmbeddingsPCA() {
        if (!this.deepModel) return;
        
        const canvas = document.getElementById('pcaChart');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Sample some items for visualization
        const sampleSize = Math.min(200, this.itemMap.size);
        const sampleIndices = [];
        for (let i = 0; i < sampleSize; i++) {
            sampleIndices.push(Math.floor(Math.random() * this.itemMap.size));
        }
        
        // Get item embeddings
        const itemTensor = tf.tensor2d(sampleIndices, [sampleSize, 1], 'int32');
        const genreTensor = tf.tensor2d(
            sampleIndices.map(idx => this.itemGenres[idx]),
            [sampleSize, this.itemGenres[0].length]
        );
        
        const embeddings = this.deepModel.itemForward(itemTensor, genreTensor);
        const embArray = embeddings.arraySync();
        
        // Simple 2D projection (PCA-like using first two dimensions)
        const points = embArray.map(emb => ({
            x: emb[0] * 100 + canvas.width / 2,
            y: emb[1] * 100 + canvas.height / 2
        }));
        
        // Draw points
        ctx.fillStyle = 'blue';
        points.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        ctx.fillStyle = 'black';
        ctx.fillText('Item Embeddings (PCA projection)', 10, 20);
        
        // Cleanup
        itemTensor.dispose();
        genreTensor.dispose();
        embeddings.dispose();
    }

    getTrainingParams() {
        return {
            maxInteractions: parseInt(document.getElementById('maxInteractions').value),
            embeddingDim: parseInt(document.getElementById('embeddingDim').value),
            hiddenDim: parseInt(document.getElementById('hiddenDim').value),
            batchSize: parseInt(document.getElementById('batchSize').value),
            epochs: parseInt(document.getElementById('epochs').value),
            learningRate: parseFloat(document.getElementById('learningRate').value),
            lossType: document.getElementById('lossType').value
        };
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
