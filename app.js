const app = {
    data: null,
    models: {},
    charts: {},
    
    async loadData() {
        this.updateStatus('Loading data files...');
        
        try {
            // Load movie data
            const itemResponse = await fetch('./data/u.item');
            const itemText = await itemResponse.text();
            
            // Load ratings data
            const dataResponse = await fetch('./data/u.data');
            const dataText = await dataResponse.text();
            
            this.parseData(itemText, dataText);
            this.updateStatus(`Data loaded: ${this.data.ratings.length} ratings, ${this.data.movies.size} movies, ${this.data.users.size} users`);
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
            console.error(error);
        }
    },
    
    parseData(itemText, dataText) {
        // Parse movie data
        const movies = new Map();
        const lines = itemText.split('\n');
        
        for (const line of lines) {
            if (!line.trim()) continue;
            const parts = line.split('|');
            if (parts.length < 24) continue;
            
            const itemId = parseInt(parts[0]);
            const title = parts[1];
            const genres = parts.slice(5, 24).map(x => parseInt(x));
            
            movies.set(itemId, {
                id: itemId,
                title: title,
                genres: genres,
                genreVector: tf.tensor1d(genres, 'float32')
            });
        }
        
        // Parse ratings data
        const ratings = [];
        const users = new Map();
        const userRatings = new Map();
        
        const ratingLines = dataText.split('\n');
        for (const line of ratingLines) {
            if (!line.trim()) continue;
            const parts = line.split('\t');
            if (parts.length < 4) continue;
            
            const userId = parseInt(parts[0]);
            const itemId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            const timestamp = parseInt(parts[3]);
            
            if (!movies.has(itemId)) continue;
            
            ratings.push({ userId, itemId, rating, timestamp });
            
            // Track unique users
            if (!users.has(userId)) {
                users.set(userId, { id: userId, ratings: [] });
            }
            
            // Track user ratings
            if (!userRatings.has(userId)) {
                userRatings.set(userId, new Set());
            }
            userRatings.get(userId).add(itemId);
            
            users.get(userId).ratings.push({ itemId, rating, timestamp });
        }
        
        // Sort user ratings by rating (desc) and timestamp (desc)
        for (const user of users.values()) {
            user.ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        }
        
        this.data = {
            movies,
            ratings,
            users,
            userRatings,
            userIdToIndex: new Map(),
            itemIdToIndex: new Map(),
            indexToUserId: [],
            indexToItemId: []
        };
        
        // Create index mappings
        let userIdx = 0;
        for (const userId of users.keys()) {
            this.data.userIdToIndex.set(userId, userIdx);
            this.data.indexToUserId.push(userId);
            userIdx++;
        }
        
        let itemIdx = 0;
        for (const itemId of movies.keys()) {
            this.data.itemIdToIndex.set(itemId, itemIdx);
            this.data.indexToItemId.push(itemId);
            itemIdx++;
        }
        
        this.initializeCharts();
    },
    
    initializeCharts() {
        // Loss chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        this.charts.loss = {
            ctx: lossCtx,
            baseline: [],
            deep: []
        };
        
        // Clear charts
        lossCtx.clearRect(0, 0, 600, 400);
        lossCtx.fillStyle = 'black';
        lossCtx.fillText('Training will populate this chart', 250, 200);
        
        // PCA chart
        const pcaCtx = document.getElementById('pcaChart').getContext('2d');
        this.charts.pca = { ctx: pcaCtx };
        pcaCtx.clearRect(0, 0, 600, 400);
        pcaCtx.fillStyle = 'black';
        pcaCtx.fillText('PCA will show after training', 250, 200);
    },
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    },
    
    async train() {
        if (!this.data) {
            this.updateStatus('Please load data first');
            return;
        }
        
        const config = this.getTrainingConfig();
        this.updateStatus('Starting training...');
        
        // Initialize models
        const genreDim = 19; // MovieLens has 19 genre bits
        
        if (config.trainBaseline) {
            this.models.baseline = new TwoTowerBaseline(
                this.data.users.size,
                this.data.movies.size,
                config.embeddingDim
            );
        }
        
        this.models.deep = new TwoTowerDeep(
            this.data.users.size,
            this.data.movies.size,
            config.embeddingDim,
            config.hiddenDim,
            genreDim
        );
        
        // Prepare training data
        const trainingData = this.prepareTrainingData(config.maxInteractions);
        
        // Train models
        this.charts.loss.baseline = [];
        this.charts.loss.deep = [];
        
        if (config.trainBaseline) {
            await this.trainModel('baseline', trainingData, config);
        }
        
        await this.trainModel('deep', trainingData, config);
        
        this.updateStatus('Training completed');
        this.drawLossChart();
        await this.drawPCA();
    },
    
    getTrainingConfig() {
        return {
            maxInteractions: parseInt(document.getElementById('maxInteractions').value),
            embeddingDim: parseInt(document.getElementById('embeddingDim').value),
            hiddenDim: parseInt(document.getElementById('hiddenDim').value),
            batchSize: parseInt(document.getElementById('batchSize').value),
            epochs: parseInt(document.getElementById('epochs').value),
            learningRate: parseFloat(document.getElementById('learningRate').value),
            lossType: document.getElementById('lossType').value,
            trainBaseline: document.getElementById('trainBaseline').value === 'yes'
        };
    },
    
    prepareTrainingData(maxInteractions) {
        const ratings = this.data.ratings
            .sort(() => Math.random() - 0.5) // Shuffle
            .slice(0, maxInteractions);
        
        return ratings.map(r => ({
            userIdx: this.data.userIdToIndex.get(r.userId),
            itemIdx: this.data.itemIdToIndex.get(r.itemId),
            rating: r.rating
        }));
    },
    
    async trainModel(modelName, trainingData, config) {
        const model = this.models[modelName];
        const optimizer = tf.train.adam(config.learningRate);
        
        const batches = [];
        for (let i = 0; i < trainingData.length; i += config.batchSize) {
            batches.push(trainingData.slice(i, i + config.batchSize));
        }
        
        for (let epoch = 0; epoch < config.epochs; epoch++) {
            let totalLoss = 0;
            
            for (const batch of batches) {
                const loss = await this.trainBatch(model, batch, optimizer, config.lossType);
                totalLoss += loss;
                
                // Update loss chart
                if (modelName === 'baseline') {
                    this.charts.loss.baseline.push(loss);
                } else {
                    this.charts.loss.deep.push(loss);
                }
                
                // Update status periodically
                if (this.charts.loss[modelName].length % 10 === 0) {
                    this.updateStatus(
                        `Training ${modelName}: Epoch ${epoch + 1}/${config.epochs}, ` +
                        `Batch ${this.charts.loss[modelName].length}, Loss: ${loss.toFixed(4)}`
                    );
                    this.drawLossChart();
                    await tf.nextFrame(); // Allow UI updates
                }
            }
            
            console.log(`${modelName} Epoch ${epoch + 1}, Average Loss: ${(totalLoss / batches.length).toFixed(4)}`);
        }
    },
    
    async trainBatch(model, batch, optimizer, lossType) {
        return tf.tidy(() => {
            const userIds = tf.tensor1d(batch.map(b => b.userIdx), 'int32');
            const itemIds = tf.tensor1d(batch.map(b => b.itemIdx), 'int32');
            
            // Get genre vectors for items
            let itemGenres = null;
            if (model instanceof TwoTowerDeep) {
                const genreArrays = batch.map(b => {
                    const itemId = this.data.indexToItemId[b.itemIdx];
                    return this.data.movies.get(itemId).genres;
                });
                itemGenres = tf.tensor2d(genreArrays, [batch.length, 19], 'float32');
            }
            
            const lossFn = () => {
                const userEmbs = model.userForward(userIds);
                const itemEmbs = model.itemForward(itemIds, itemGenres);
                
                if (lossType === 'softmax') {
                    return this.inBatchSoftmaxLoss(userEmbs, itemEmbs);
                } else {
                    return this.bprLoss(userEmbs, itemEmbs, batch.length);
                }
            };
            
            const loss = optimizer.minimize(lossFn, true);
            return loss ? loss.dataSync()[0] : 0;
        });
    },
    
    inBatchSoftmaxLoss(userEmbs, itemEmbs) {
        // Normalize embeddings
        const u = tf.l2Normalize(userEmbs, -1);
        const v = tf.l2Normalize(itemEmbs, -1);
        
        // Compute similarity matrix
        const logits = tf.matMul(u, v, false, true); // [B, B]
        
        // Labels are diagonal (each user's positive item is on diagonal)
        const labels = tf.oneHot(tf.range(0, logits.shape[0]), logits.shape[1]);
        
        return tf.losses.softmaxCrossEntropy(labels, logits);
    },
    
    bprLoss(userEmbs, itemEmbs, batchSize) {
        // Sample negatives
        const negItemIndices = tf.randomUniform([batchSize], 0, this.data.movies.size, 'int32');
        const negItemEmbs = tf.l2Normalize(tf.gather(this.models.deep.itemIdEmbedding, negItemIndices), -1);
        
        const posScores = tf.sum(tf.l2Normalize(userEmbs, -1).mul(tf.l2Normalize(itemEmbs, -1)), -1);
        const negScores = tf.sum(tf.l2Normalize(userEmbs, -1).mul(negItemEmbs), -1);
        
        const diff = posScores.sub(negScores);
        const losses = tf.log(tf.sigmoid(diff));
        
        return tf.neg(tf.mean(losses));
    },
    
    drawLossChart() {
        const ctx = this.charts.loss.ctx;
        const width = 600;
        const height = 400;
        const padding = 50;
        
        ctx.clearRect(0, 0, width, height);
        
        if (this.charts.loss.baseline.length === 0 && this.charts.loss.deep.length === 0) {
            return;
        }
        
        // Find max loss for scaling
        const allLosses = [...this.charts.loss.baseline, ...this.charts.loss.deep];
        const maxLoss = Math.max(...allLosses);
        const minLoss = Math.min(...allLosses);
        
        // Draw axes
        ctx.strokeStyle = '#000';
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        // Draw baseline loss
        if (this.charts.loss.baseline.length > 0) {
            ctx.strokeStyle = 'green';
            ctx.beginPath();
            this.charts.loss.baseline.forEach((loss, i) => {
                const x = padding + (i / this.charts.loss.baseline.length) * (width - 2 * padding);
                const y = height - padding - ((loss - minLoss) / (maxLoss - minLoss)) * (height - 2 * padding);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        // Draw deep loss
        if (this.charts.loss.deep.length > 0) {
            ctx.strokeStyle = 'red';
            ctx.beginPath();
            this.charts.loss.deep.forEach((loss, i) => {
                const x = padding + (i / this.charts.loss.deep.length) * (width - 2 * padding);
                const y = height - padding - ((loss - minLoss) / (maxLoss - minLoss)) * (height - 2 * padding);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        // Draw legend
        ctx.fillStyle = 'black';
        ctx.fillText(`Loss Range: ${minLoss.toFixed(3)} - ${maxLoss.toFixed(3)}`, padding, padding - 10);
        
        if (this.charts.loss.baseline.length > 0) {
            ctx.fillStyle = 'green';
            ctx.fillText('Baseline', width - 100, padding - 10);
        }
        
        if (this.charts.loss.deep.length > 0) {
            ctx.fillStyle = 'red';
            ctx.fillText('Deep', width - 100, padding + 10);
        }
    },
    
    async drawPCA() {
        if (!this.models.deep) return;
        
        const ctx = this.charts.pca.ctx;
        const width = 600;
        const height = 400;
        const padding = 50;
        
        ctx.clearRect(0, 0, width, height);
        
        // Sample some items for PCA
        const sampleSize = Math.min(200, this.data.movies.size);
        const sampleIndices = Array.from({length: sampleSize}, (_, i) => 
            Math.floor(i * this.data.movies.size / sampleSize)
        );
        
        const itemIds = tf.tensor1d(sampleIndices, 'int32');
        const genreArrays = sampleIndices.map(idx => {
            const itemId = this.data.indexToItemId[idx];
            return this.data.movies.get(itemId).genres;
        });
        const itemGenres = tf.tensor2d(genreArrays, [sampleSize, 19], 'float32');
        
        const embeddings = this.models.deep.itemForward(itemIds, itemGenres);
        const embeddingArray = await embeddings.array();
        
        // Simple 2D projection (simplified PCA)
        const projected = this.simpleProjection(embeddingArray);
        
        // Draw points
        ctx.fillStyle = 'blue';
        projected.forEach(point => {
            const x = padding + (point[0] + 1) * 0.5 * (width - 2 * padding);
            const y = padding + (point[1] + 1) * 0.5 * (height - 2 * padding);
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        ctx.fillStyle = 'black';
        ctx.fillText('Item Embeddings (2D Projection)', width / 2 - 80, padding - 10);
        
        embeddings.dispose();
        itemIds.dispose();
        itemGenres.dispose();
    },
    
    simpleProjection(embeddings) {
        // Simple projection using first two dimensions (for demo purposes)
        // In a real PCA, we'd compute eigenvectors of covariance matrix
        return embeddings.map(emb => {
            // Use first two dimensions, normalized
            const x = emb[0] || 0;
            const y = emb[1] || 0;
            const norm = Math.sqrt(x * x + y * y) || 1;
            return [x / norm, y / norm];
        });
    },
    
    async test() {
        if (!this.models.deep) {
            this.updateStatus('Please train models first');
            return;
        }
        
        this.updateStatus('Generating recommendations...');
        
        // Find a user with at least 20 ratings
        const eligibleUsers = Array.from(this.data.users.entries())
            .filter(([_, user]) => user.ratings.length >= 20);
        
        if (eligibleUsers.length === 0) {
            this.updateStatus('No users with sufficient ratings found');
            return;
        }
        
        const randomUser = eligibleUsers[Math.floor(Math.random() * eligibleUsers.length)];
        const [userId, userData] = randomUser;
        const userIdx = this.data.userIdToIndex.get(userId);
        
        // Get user's rated items
        const ratedItems = new Set(userData.ratings.map(r => r.itemId));
        
        // Generate recommendations
        const baselineRecs = await this.getRecommendations('baseline', userIdx, ratedItems);
        const deepRecs = await this.getRecommendations('deep', userIdx, ratedItems);
        
        this.renderResults(userData, baselineRecs, deepRecs);
        this.updateStatus(`Recommendations generated for user ${userId}`);
    },
    
    async getRecommendations(modelName, userIdx, ratedItems) {
        if (!this.models[modelName]) return [];
        
        return tf.tidy(() => {
            const userTensor = tf.tensor1d([userIdx], 'int32');
            const userEmb = this.models[modelName].userForward(userTensor);
            
            // Score all items
            const allItemIndices = tf.range(0, this.data.movies.size, 1, 'int32');
            
            let allItemEmbs;
            if (modelName === 'baseline') {
                allItemEmbs = this.models.baseline.itemForward(allItemIndices);
            } else {
                const genreArrays = this.data.indexToItemId.map(itemId => 
                    this.data.movies.get(itemId).genres
                );
                const itemGenres = tf.tensor2d(genreArrays, [this.data.movies.size, 19], 'float32');
                allItemEmbs = this.models.deep.itemForward(allItemIndices, itemGenres);
                itemGenres.dispose();
            }
            
            const scores = this.models[modelName].score(userEmb, allItemEmbs);
            const scoreArray = scores.dataSync();
            
            // Get top recommendations (excluding rated items)
            const recommendations = [];
            for (let i = 0; i < scoreArray.length; i++) {
                const itemId = this.data.indexToItemId[i];
                if (!ratedItems.has(itemId)) {
                    recommendations.push({
                        itemId: itemId,
                        score: scoreArray[i],
                        movie: this.data.movies.get(itemId)
                    });
                }
            }
            
            recommendations.sort((a, b) => b.score - a.score);
            return recommendations.slice(0, 10);
        });
    },
    
    renderResults(userData, baselineRecs, deepRecs) {
        const container = document.getElementById('results');
        
        const topRated = userData.ratings.slice(0, 10).map(r => ({
            movie: this.data.movies.get(r.itemId),
            rating: r.rating
        }));
        
        let html = `<h2>Recommendations for User ${userData.id}</h2>`;
        html += `<table>
            <tr>
                <th>Top 10 Rated (Historical)</th>
                <th>Top 10 Recommended (Baseline)</th>
                <th>Top 10 Recommended (Deep)</th>
            </tr>`;
        
        for (let i = 0; i < 10; i++) {
            html += '<tr>';
            
            // Historical ratings
            if (topRated[i]) {
                html += `<td>${topRated[i].movie.title} (Rating: ${topRated[i].rating})</td>`;
            } else {
                html += '<td>-</td>';
            }
            
            // Baseline recommendations
            if (baselineRecs[i]) {
                html += `<td>${baselineRecs[i].movie.title} (Score: ${baselineRecs[i].score.toFixed(4)})</td>`;
            } else {
                html += '<td>-</td>';
            }
            
            // Deep recommendations
            if (deepRecs[i]) {
                html += `<td>${deepRecs[i].movie.title} (Score: ${deepRecs[i].score.toFixed(4)})</td>`;
            } else {
                html += '<td>-</td>';
            }
            
            html += '</tr>';
        }
        
        html += '</table>';
        container.innerHTML = html;
    }
};
