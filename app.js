// Main application for Two-Tower Movie Recommender
class MovieRecommenderApp {
    constructor() {
        this.data = null;
        this.baselineModel = null;
        this.deepModel = null;
        this.isTraining = false;
        this.lossHistory = { baseline: [], deep: [] };
        
        // DOM elements
        this.statusEl = document.getElementById('status');
        this.loadBtn = document.getElementById('loadData');
        this.trainBtn = document.getElementById('train');
        this.testBtn = document.getElementById('test');
        this.lossCanvas = document.getElementById('lossChart');
        this.pcaCanvas = document.getElementById('pcaChart');
        this.comparisonEl = document.getElementById('comparison');
        
        // Initialize charts
        this.lossCtx = this.lossCanvas.getContext('2d');
        this.pcaCtx = this.pcaCanvas.getContext('2d');
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.loadBtn.addEventListener('click', () => this.loadData());
        this.trainBtn.addEventListener('click', () => this.trainModels());
        this.testBtn.addEventListener('click', () => this.testModels());
    }

    updateStatus(message) {
        this.statusEl.textContent = message;
        console.log(message);
    }

    async loadData() {
        try {
            this.updateStatus('Loading MovieLens 100K data...');
            this.loadBtn.disabled = true;

            // Load items data
            const itemsResponse = await fetch('./u.item');
            if (!itemsResponse.ok) throw new Error('Failed to load u.item');
            const itemsText = await itemsResponse.text();
            
            // Load ratings data
            const ratingsResponse = await fetch('./u.data');
            if (!ratingsResponse.ok) throw new Error('Failed to load u.data');
            const ratingsText = await ratingsResponse.text();

            this.data = this.parseData(itemsText, ratingsText);
            this.updateStatus(`Data loaded: ${this.data.ratings.length} ratings, ${this.data.items.size} movies, ${this.data.users.size} users`);
            
            this.trainBtn.disabled = false;
            this.testBtn.disabled = false;

        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
            this.loadBtn.disabled = false;
        }
    }

    parseData(itemsText, ratingsText) {
        const items = new Map();
        const users = new Map();
        const ratings = [];
        const userRatings = new Map();

        // Parse items (movies)
        const itemLines = itemsText.split('\n');
        for (const line of itemLines) {
            if (!line.trim()) continue;
            
            const parts = line.split('|');
            if (parts.length < 24) continue; // Should have id, title, release_date, video_release_date, imdb_url, and 19 genres
            
            const itemId = parseInt(parts[0]);
            const title = parts[1];
            const genres = parts.slice(5, 24).map(g => parseInt(g)); // 19 genre flags
            
            // Extract year from title if present (format: "Title (YYYY)")
            let year = null;
            const yearMatch = title.match(/\((\d{4})\)$/);
            if (yearMatch) {
                year = parseInt(yearMatch[1]);
            }
            
            items.set(itemId, {
                id: itemId,
                title: title,
                year: year,
                genres: genres,
                genreVector: tf.tensor1d(genres, 'float32')
            });
        }

        // Parse ratings
        const maxInteractions = parseInt(document.getElementById('maxInteractions').value);
        const ratingLines = ratingsText.split('\n').slice(0, maxInteractions);
        
        for (const line of ratingLines) {
            if (!line.trim()) continue;
            
            const parts = line.split('\t');
            if (parts.length < 4) continue;
            
            const userId = parseInt(parts[0]);
            const itemId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            const timestamp = parseInt(parts[3]);
            
            if (!items.has(itemId)) continue; // Skip if item not found
            
            ratings.push({
                userId,
                itemId,
                rating,
                timestamp
            });

            // Track user ratings
            if (!userRatings.has(userId)) {
                userRatings.set(userId, []);
            }
            userRatings.get(userId).push({ itemId, rating, timestamp });
            
            // Add to users map
            if (!users.has(userId)) {
                users.set(userId, { id: userId, ratings: [] });
            }
        }

        // Create index mappings
        const userIdToIndex = new Map();
        const itemIdToIndex = new Map();
        const indexToUserId = [];
        const indexToItemId = [];

        // User indices
        let userIndex = 0;
        for (const userId of users.keys()) {
            userIdToIndex.set(userId, userIndex);
            indexToUserId.push(userId);
            userIndex++;
        }

        // Item indices
        let itemIndex = 0;
        for (const itemId of items.keys()) {
            itemIdToIndex.set(itemId, itemIndex);
            indexToItemId.push(itemId);
            itemIndex++;
        }

        return {
            items,
            users,
            ratings,
            userRatings,
            userIdToIndex,
            itemIdToIndex,
            indexToUserId,
            indexToItemId,
            numUsers: users.size,
            numItems: items.size,
            genreDim: 19 // 19 genres in MovieLens
        };
    }

    async trainModels() {
        if (!this.data) {
            this.updateStatus('Please load data first');
            return;
        }

        try {
            this.isTraining = true;
            this.trainBtn.disabled = true;
            this.updateStatus('Training models...');

            const config = this.getTrainingConfig();
            this.lossHistory = { baseline: [], deep: [] };

            // Initialize models
            const trainBaseline = document.getElementById('trainBaseline').value === 'yes';
            
            if (trainBaseline) {
                this.baselineModel = new TwoTowerBaseline(
                    this.data.numUsers, 
                    this.data.numItems, 
                    config.embeddingDim
                );
            }
            
            this.deepModel = new TwoTowerDeep(
                this.data.numUsers,
                this.data.numItems,
                config.embeddingDim,
                config.hiddenDim,
                this.data.genreDim
            );

            // Train models
            if (trainBaseline) {
                await this.trainModel(this.baselineModel, 'baseline', config);
            }
            await this.trainModel(this.deepModel, 'deep', config);

            // Draw PCA visualization
            await this.drawItemEmbeddingsPCA();

            this.updateStatus('Training completed! Click Test to see recommendations.');
            this.trainBtn.disabled = false;
            this.isTraining = false;

        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`);
            this.trainBtn.disabled = false;
            this.isTraining = false;
        }
    }

    async trainModel(model, modelName, config) {
        const optimizer = tf.train.adam(config.learningRate);
        const trainer = new TwoTowerTrainer(model, optimizer, config.lossType);
        
        const numBatches = Math.ceil(this.data.ratings.length / config.batchSize);
        
        for (let epoch = 0; epoch < config.epochs; epoch++) {
            let epochLoss = 0;
            let batches = 0;
            
            // Shuffle ratings for each epoch
            const shuffledRatings = [...this.data.ratings];
            this.shuffleArray(shuffledRatings);
            
            for (let i = 0; i < shuffledRatings.length; i += config.batchSize) {
                const batchRatings = shuffledRatings.slice(i, i + config.batchSize);
                
                // Prepare batch tensors
                const userBatch = [];
                const itemBatch = [];
                const genresBatch = [];
                
                for (const rating of batchRatings) {
                    userBatch.push(this.data.userIdToIndex.get(rating.userId));
                    itemBatch.push(this.data.itemIdToIndex.get(rating.itemId));
                    
                    if (modelName === 'deep') {
                        const item = this.data.items.get(rating.itemId);
                        genresBatch.push(item.genres);
                    }
                }
                
                const userTensor = tf.tensor2d(userBatch, [batchRatings.length, 1], 'int32');
                const itemTensor = tf.tensor2d(itemBatch, [batchRatings.length, 1], 'int32');
                let genresTensor = null;
                
                if (modelName === 'deep' && genresBatch.length > 0) {
                    genresTensor = tf.tensor2d(genresBatch, [batchRatings.length, this.data.genreDim], 'float32');
                }
                
                const loss = await trainer.trainStep(userTensor, itemTensor, genresTensor);
                
                epochLoss += loss;
                batches++;
                
                // Update loss history and chart
                this.lossHistory[modelName].push(loss);
                this.updateLossChart();
                
                // Clean up tensors
                tf.dispose([userTensor, itemTensor]);
                if (genresTensor) tf.dispose(genresTensor);
                
                if (batches % 10 === 0) {
                    this.updateStatus(`Training ${modelName} - Epoch ${epoch + 1}/${config.epochs}, Batch ${batches}/${numBatches}, Loss: ${loss.toFixed(4)}`);
                }
                
                await tf.nextFrame(); // Allow UI updates
            }
            
            epochLoss /= batches;
            console.log(`${modelName} Epoch ${epoch + 1}, Average Loss: ${epochLoss.toFixed(4)}`);
        }
    }

    async testModels() {
        if (!this.data || (!this.baselineModel && !this.deepModel)) {
            this.updateStatus('Please train models first');
            return;
        }

        try {
            this.updateStatus('Generating recommendations...');

            // Find a user with at least 20 ratings
            const eligibleUsers = Array.from(this.data.userRatings.entries())
                .filter(([_, ratings]) => ratings.length >= 20)
                .map(([userId]) => userId);
            
            if (eligibleUsers.length === 0) {
                this.updateStatus('No users with sufficient ratings found');
                return;
            }

            const randomUser = eligibleUsers[Math.floor(Math.random() * eligibleUsers.length)];
            const userRatings = this.data.userRatings.get(randomUser);
            
            // Get top 10 rated movies (historical)
            const topRated = userRatings
                .sort((a, b) => b.rating - a.rating || b.timestamp - a.timestamp)
                .slice(0, 10)
                .map(r => ({
                    itemId: r.itemId,
                    rating: r.rating,
                    ...this.data.items.get(r.itemId)
                }));

            // Generate recommendations (exclude already rated items)
            const ratedItemIds = new Set(userRatings.map(r => r.itemId));
            const unratedItems = Array.from(this.data.items.values())
                .filter(item => !ratedItemIds.has(item.id));

            // Get baseline recommendations
            let baselineRecs = [];
            if (this.baselineModel) {
                baselineRecs = await this.generateRecommendations(
                    this.baselineModel, randomUser, unratedItems, 'baseline'
                );
            }

            // Get deep recommendations
            const deepRecs = await this.generateRecommendations(
                this.deepModel, randomUser, unratedItems, 'deep'
            );

            // Render comparison table
            this.renderComparisonTable(randomUser, topRated, baselineRecs, deepRecs);
            
            this.updateStatus(`Recommendations generated for user ${randomUser}`);

        } catch (error) {
            this.updateStatus(`Testing error: ${error.message}`);
        }
    }

    async generateRecommendations(model, userId, candidateItems, modelType) {
        const userIndex = this.data.userIdToIndex.get(userId);
        const userTensor = tf.tensor2d([userIndex], [1, 1], 'int32');
        
        try {
            const scores = [];
            
            // Score items in batches to avoid memory issues
            const batchSize = 100;
            for (let i = 0; i < candidateItems.length; i += batchSize) {
                const batchItems = candidateItems.slice(i, i + batchSize);
                const itemIndices = batchItems.map(item => this.data.itemIdToIndex.get(item.id));
                const itemTensor = tf.tensor2d(itemIndices, [batchItems.length, 1], 'int32');
                
                let predictions;
                if (modelType === 'deep') {
                    const genreVectors = batchItems.map(item => item.genres);
                    const genresTensor = tf.tensor2d(genreVectors, [batchItems.length, this.data.genreDim], 'float32');
                    predictions = model.predict(userTensor, itemTensor, genresTensor);
                    tf.dispose(genresTensor);
                } else {
                    predictions = model.predict(userTensor, itemTensor);
                }
                
                const scoreValues = await predictions.data();
                
                for (let j = 0; j < batchItems.length; j++) {
                    scores.push({
                        ...batchItems[j],
                        score: scoreValues[j]
                    });
                }
                
                tf.dispose([itemTensor, predictions]);
                await tf.nextFrame();
            }
            
            // Return top 10 by score
            return scores
                .sort((a, b) => b.score - a.score)
                .slice(0, 10);
                
        } finally {
            tf.dispose(userTensor);
        }
    }

    renderComparisonTable(userId, topRated, baselineRecs, deepRecs) {
        let html = `
            <h3>User ${userId} - Recommendation Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Top 10 Rated (Historical)</th>
                        ${this.baselineModel ? '<th>Top 10 Recommended (Baseline)</th>' : ''}
                        <th>Top 10 Recommended (Deep)</th>
                    </tr>
                </thead>
                <tbody>
        `;

        const maxRows = Math.max(topRated.length, baselineRecs.length, deepRecs.length);
        
        for (let i = 0; i < maxRows; i++) {
            html += '<tr>';
            
            // Historical ratings column
            if (i < topRated.length) {
                const movie = topRated[i];
                html += `
                    <td>
                        <div class="movie-title">${this.escapeHtml(movie.title)}</div>
                        <div class="movie-genres">Rating: ${movie.rating}/5 | ${this.getGenreNames(movie.genres)}</div>
                    </td>
                `;
            } else {
                html += '<td></td>';
            }
            
            // Baseline recommendations column
            if (this.baselineModel) {
                if (i < baselineRecs.length) {
                    const movie = baselineRecs[i];
                    html += `
                        <td>
                            <div class="movie-title">${this.escapeHtml(movie.title)}</div>
                            <div class="movie-genres">Score: ${movie.score.toFixed(4)} | ${this.getGenreNames(movie.genres)}</div>
                        </td>
                    `;
                } else {
                    html += '<td></td>';
                }
            }
            
            // Deep recommendations column
            if (i < deepRecs.length) {
                const movie = deepRecs[i];
                html += `
                    <td>
                        <div class="movie-title">${this.escapeHtml(movie.title)}</div>
                        <div class="movie-genres">Score: ${movie.score.toFixed(4)} | ${this.getGenreNames(movie.genres)}</div>
                    </td>
                `;
            } else {
                html += '<td></td>';
            }
            
            html += '</tr>';
        }
        
        html += '</tbody></table>';
        this.comparisonEl.innerHTML = html;
    }

    async drawItemEmbeddingsPCA() {
        if (!this.deepModel) return;

        try {
            // Get item embeddings from deep model
            const allItemIndices = Array.from({ length: this.data.numItems }, (_, i) => i);
            const itemTensor = tf.tensor2d(allItemIndices, [this.data.numItems, 1], 'int32');
            
            // Get genre vectors for all items
            const genreVectors = [];
            for (let i = 0; i < this.data.numItems; i++) {
                const itemId = this.data.indexToItemId[i];
                const item = this.data.items.get(itemId);
                genreVectors.push(item.genres);
            }
            const genresTensor = tf.tensor2d(genreVectors, [this.data.numItems, this.data.genreDim], 'float32');
            
            const itemEmbs = this.deepModel.itemForward(itemTensor, genresTensor);
            
            // Compute PCA to 2D
            const pcaResult = await this.computePCA(await itemEmbs.array(), 2);
            
            // Draw scatter plot
            this.drawScatterPlot(pcaResult);
            
            tf.dispose([itemTensor, genresTensor, itemEmbs]);
            
        } catch (error) {
            console.error('PCA error:', error);
        }
    }

    async computePCA(data, components = 2) {
        return tf.tidy(() => {
            const X = tf.tensor2d(data);
            
            // Center the data
            const mean = tf.mean(X, 0);
            const centered = X.sub(mean);
            
            // Compute covariance matrix
            const covariance = tf.matMul(centered.transpose(), centered).div(X.shape[0] - 1);
            
            // Compute eigenvalues and eigenvectors
            const [eigenvalues, eigenvectors] = tf.linalg.eigh(covariance);
            
            // Get top components
            const topIndices = Array.from({ length: components }, (_, i) => eigenvectors.shape[1] - 1 - i);
            const topVectors = tf.gather(eigenvectors, topIndices, 1).transpose();
            
            // Project data
            const projected = tf.matMul(centered, topVectors);
            
            return projected.arraySync();
        });
    }

    drawScatterPlot(points) {
        const width = this.pcaCanvas.width;
        const height = this.pcaCanvas.height;
        const padding = 40;
        
        // Clear canvas
        this.pcaCtx.clearRect(0, 0, width, height);
        
        // Find bounds
        const xValues = points.map(p => p[0]);
        const yValues = points.map(p => p[1]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        // Scale function
        const scaleX = (x) => padding + (x - xMin) / (xMax - xMin) * (width - 2 * padding);
        const scaleY = (y) => height - padding - (y - yMin) / (yMax - yMin) * (height - 2 * padding);
        
        // Draw points
        this.pcaCtx.fillStyle = 'rgba(74, 144, 226, 0.6)';
        for (const point of points) {
            const x = scaleX(point[0]);
            const y = scaleY(point[1]);
            this.pcaCtx.beginPath();
            this.pcaCtx.arc(x, y, 3, 0, 2 * Math.PI);
            this.pcaCtx.fill();
        }
        
        // Draw axes
        this.pcaCtx.strokeStyle = '#ccc';
        this.pcaCtx.lineWidth = 1;
        this.pcaCtx.beginPath();
        this.pcaCtx.moveTo(padding, padding);
        this.pcaCtx.lineTo(padding, height - padding);
        this.pcaCtx.lineTo(width - padding, height - padding);
        this.pcaCtx.stroke();
        
        // Labels
        this.pcaCtx.fillStyle = '#333';
        this.pcaCtx.font = '12px Arial';
        this.pcaCtx.fillText('PCA Component 1', width / 2,
