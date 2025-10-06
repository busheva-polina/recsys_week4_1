// app.js
class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map();
        this.itemMap = new Map();
        this.reverseUserMap = new Map();
        this.reverseItemMap = new Map();
        this.userRatedItems = new Map();
        this.userTopRated = new Map();
        this.genres = [];
        this.genreMap = new Map();
        
        this.basicModel = null;
        this.dlModel = null;
        this.isDataLoaded = false;
        
        this.setupEventListeners();
        this.updateStatus('Please click "Load Data" to start');
    }

    setupEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainBasic').addEventListener('click', () => this.trainModel('basic'));
        document.getElementById('trainDL').addEventListener('click', () => this.trainModel('dl'));
        document.getElementById('test').addEventListener('click', () => this.testRecommendations());
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }

    async loadData() {
        try {
            this.updateStatus('Loading data...');
            
            // Load interactions
            const dataResponse = await fetch('data/u.data');
            const dataText = await dataResponse.text();
            const dataLines = dataText.trim().split('\n');
            
            this.interactions = dataLines.map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });

            // Load items and parse genres
            const itemResponse = await fetch('data/u.item');
            const itemText = await itemResponse.text();
            const itemLines = itemText.trim().split('\n');
            
            // Genre list from MovieLens documentation
            this.genres = [
                "Unknown", "Action", "Adventure", "Animation", "Children's", 
                "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                "Sci-Fi", "Thriller", "War", "Western"
            ];
            
            // Create genre mapping
            this.genres.forEach((genre, index) => {
                this.genreMap.set(index, genre);
            });

            itemLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const releaseDate = parts[2];
                const genreFlags = parts.slice(5, 24).map(flag => parseInt(flag));
                
                // Extract year from title (format: "Movie Title (YYYY)")
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                
                // Get genre names from flags
                const itemGenres = genreFlags
                    .map((flag, idx) => flag === 1 ? this.genres[idx] : null)
                    .filter(genre => genre !== null);
                
                this.items.set(itemId, {
                    title: title.replace(/\s*\(\d{4}\)$/, ''), // Remove year from title
                    year: year,
                    genres: itemGenres,
                    genreFlags: genreFlags
                });
            });

            this.buildMappings();
            this.buildUserStats();
            
            this.isDataLoaded = true;
            document.getElementById('trainBasic').disabled = false;
            document.getElementById('trainDL').disabled = false;
            
            this.updateStatus(`Data loaded: ${this.interactions.length} interactions, ${this.items.size} movies, ${this.userMap.size} users`);
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }

    buildMappings() {
        // Build user and item mappings
        const uniqueUsers = [...new Set(this.interactions.map(i => i.userId))].sort((a, b) => a - b);
        const uniqueItems = [...new Set(this.interactions.map(i => i.itemId))].sort((a, b) => a - b);
        
        uniqueUsers.forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });
        
        uniqueItems.forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
    }

    buildUserStats() {
        // Build user ratings data
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            const itemId = interaction.itemId;
            
            if (!this.userRatedItems.has(userId)) {
                this.userRatedItems.set(userId, []);
            }
            this.userRatedItems.get(userId).push({
                itemId: itemId,
                rating: interaction.rating,
                timestamp: interaction.timestamp
            });
        });

        // Build top rated items for each user
        this.userRatedItems.forEach((ratings, userId) => {
            const sorted = ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
            this.userTopRated.set(userId, sorted.slice(0, 10));
        });
    }

    async trainModel(modelType) {
        if (!this.isDataLoaded) {
            this.updateStatus('Please load data first');
            return;
        }

        try {
            const config = {
                embeddingDim: 32,
                learningRate: 0.001,
                epochs: 10,
                batchSize: 512
            };

            if (modelType === 'basic') {
                this.updateStatus('Training Basic Model...');
                this.basicModel = new TwoTowerModel(
                    this.userMap.size,
                    this.itemMap.size,
                    config.embeddingDim,
                    false // no deep learning
                );
                await this.trainModelInternal(this.basicModel, config, 'basic');
            } else {
                this.updateStatus('Training Deep Learning Model...');
                this.dlModel = new TwoTowerModel(
                    this.userMap.size,
                    this.itemMap.size,
                    config.embeddingDim,
                    true, // use deep learning
                    this.genres.length
                );
                await this.trainModelInternal(this.dlModel, config, 'dl');
            }

            document.getElementById('test').disabled = false;
            
        } catch (error) {
            this.updateStatus(`Training error: ${error.message}`);
        }
    }

    async trainModelInternal(model, config, modelType) {
        const lossChartId = modelType === 'basic' ? 'lossChartBasic' : 'lossChartDL';
        const lossCtx = document.getElementById(lossChartId).getContext('2d');
        this.setupLossChart(lossCtx);
        
        const losses = [];
        
        for (let epoch = 0; epoch < config.epochs; epoch++) {
            this.updateStatus(`Training ${modelType.toUpperCase()} Model - Epoch ${epoch + 1}/${config.epochs}`);
            
            const epochLoss = await model.trainEpoch(
                this.interactions,
                this.userMap,
                this.itemMap,
                config.batchSize,
                this.items
            );
            
            losses.push(epochLoss);
            this.updateLossChart(lossCtx, losses, modelType);
            
            // Force UI update
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        
        this.updateStatus(`${modelType.toUpperCase()} Model training completed`);
        
        // Create embedding visualization
        await this.visualizeEmbeddings(model, modelType);
    }

    setupLossChart(ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        ctx.strokeStyle = '#007cba';
        ctx.lineWidth = 2;
        ctx.beginPath();
    }

    updateLossChart(ctx, losses, modelType) {
        const colors = {
            basic: '#007cba',
            dl: '#dc3545'
        };
        
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        // Draw grid
        ctx.strokeStyle = '#e9ecef';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = ctx.canvas.height - (i * ctx.canvas.height / 5);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(ctx.canvas.width, y);
            ctx.stroke();
        }
        
        // Draw loss line
        ctx.strokeStyle = colors[modelType];
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const maxLoss = Math.max(...losses);
        const minLoss = Math.min(...losses);
        const range = maxLoss - minLoss || 1;
        
        losses.forEach((loss, index) => {
            const x = (index / (losses.length - 1 || 1)) * ctx.canvas.width;
            const y = ctx.canvas.height - ((loss - minLoss) / range) * ctx.canvas.height * 0.9;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
        
        // Draw points
        ctx.fillStyle = colors[modelType];
        losses.forEach((loss, index) => {
            const x = (index / (losses.length - 1 || 1)) * ctx.canvas.width;
            const y = ctx.canvas.height - ((loss - minLoss) / range) * ctx.canvas.height * 0.9;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    async visualizeEmbeddings(model, modelType) {
        const canvasId = modelType === 'basic' ? 'projectionBasic' : 'projectionDL';
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        // Sample items for visualization
        const sampleSize = Math.min(200, this.itemMap.size);
        const sampleItems = Array.from(this.itemMap.values())
            .sort(() => Math.random() - 0.5)
            .slice(0, sampleSize);
        
        const itemEmbeddings = await model.getItemEmbeddings(sampleItems);
        
        // Simple PCA projection (2D)
        const projected = this.simplePCA(itemEmbeddings, 2);
        
        // Draw scatter plot
        this.drawEmbeddingProjection(ctx, projected, sampleItems, modelType);
    }

    simplePCA(embeddings, components = 2) {
        // Center the data
        const mean = tf.mean(embeddings, 0);
        const centered = embeddings.sub(mean);
        
        // Compute covariance matrix
        const cov = tf.matMul(centered.transpose(), centered).div(embeddings.shape[0] - 1);
        
        // Simple power iteration for top components
        const result = [];
        let remaining = centered;
        
        for (let i = 0; i < components; i++) {
            // Power iteration
            let v = tf.randomNormal([embeddings.shape[1], 1]);
            for (let iter = 0; iter < 10; iter++) {
                v = tf.matMul(cov, v);
                v = v.div(tf.norm(v));
            }
            
            // Project data
            const projection = tf.matMul(remaining, v).flatten();
            result.push(projection.arraySync());
            
            // Deflate
            const projectedComponent = tf.matMul(projection.expandDims(1), v.transpose());
            remaining = remaining.sub(projectedComponent);
        }
        
        // Combine components
        const projectedData = [];
        for (let i = 0; i < embeddings.shape[0]; i++) {
            projectedData.push([result[0][i], result[1][i]]);
        }
        
        return projectedData;
    }

    drawEmbeddingProjection(ctx, points, itemIndices, modelType) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        // Find bounds
        const xValues = points.map(p => p[0]);
        const yValues = points.map(p => p[1]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        const scaleX = ctx.canvas.width / (xMax - xMin || 1);
        const scaleY = ctx.canvas.height / (yMax - yMin || 1);
        const scale = Math.min(scaleX, scaleY) * 0.8;
        
        const offsetX = (ctx.canvas.width - (xMax - xMin) * scale) / 2;
        const offsetY = (ctx.canvas.height - (yMax - yMin) * scale) / 2;
        
        // Draw points
        points.forEach((point, index) => {
            const x = offsetX + (point[0] - xMin) * scale;
            const y = offsetY + (point[1] - yMin) * scale;
            
            ctx.fillStyle = modelType === 'basic' ? '#007cba' : '#dc3545';
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Add title
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText(`Item Embeddings (${points.length} samples)`, 10, 20);
    }

    async testRecommendations() {
        if (!this.basicModel || !this.dlModel) {
            this.updateStatus('Please train both models first');
            return;
        }

        try {
            // Find users with at least 20 ratings
            const qualifiedUsers = Array.from(this.userRatedItems.entries())
                .filter(([_, ratings]) => ratings.length >= 20)
                .map(([userId]) => userId);
            
            if (qualifiedUsers.length === 0) {
                this.updateStatus('No users with 20+ ratings found');
                return;
            }
            
            // Pick random user
            const randomUser = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
            const userIndex = this.userMap.get(randomUser);
            
            this.updateStatus(`Testing recommendations for user ${randomUser}...`);
            
            // Get recommendations from both models
            const basicRecs = await this.getRecommendations(this.basicModel, userIndex, randomUser);
            const dlRecs = await this.getRecommendations(this.dlModel, userIndex, randomUser);
            
            // Display results
            this.displayResults(basicRecs, dlRecs, randomUser);
            
        } catch (error) {
            this.updateStatus(`Testing error: ${error.message}`);
        }
    }

    async getRecommendations(model, userIndex, userId) {
        const userEmb = await model.getUserEmbedding(userIndex);
        const allItemIndices = Array.from(this.itemMap.values());
        
        // Get scores for all items
        const scores = await model.getScoresForAllItems(userEmb, allItemIndices);
        
        // Get user's rated items
        const ratedItems = this.userRatedItems.get(userId).map(r => r.itemId);
        const ratedItemIndices = ratedItems.map(itemId => this.itemMap.get(itemId));
        
        // Filter out rated items and get top 10
        const recommendations = [];
        for (let i = 0; i < scores.length; i++) {
            const itemIndex = allItemIndices[i];
            const itemId = this.reverseItemMap.get(itemIndex);
            
            if (!ratedItemIndices.includes(itemIndex)) {
                recommendations.push({
                    itemId: itemId,
                    score: scores[i],
                    itemIndex: itemIndex
                });
            }
        }
        
        return recommendations
            .sort((a, b) => b.score - a.score)
            .slice(0, 10)
            .map(rec => {
                const item = this.items.get(rec.itemId);
                return {
                    title: item.title,
                    year: item.year,
                    score: rec.score.toFixed(4),
                    genres: item.genres.join(', ')
                };
            });
    }

    displayResults(basicRecs, dlRecs, userId) {
        const userTopRated = this.userTopRated.get(userId).slice(0, 10).map(rating => {
            const item = this.items.get(rating.itemId);
            return {
                title: item.title,
                year: item.year,
                rating: rating.rating,
                genres: item.genres.join(', ')
            };
        });

        // Display basic model results
        const basicResultsDiv = document.getElementById('resultsBasic');
        basicResultsDiv.innerHTML = this.createComparisonTable(userTopRated, basicRecs, 'Basic Model');
        
        // Display DL model results
        const dlResultsDiv = document.getElementById('resultsDL');
        dlResultsDiv.innerHTML = this.createComparisonTable(userTopRated, dlRecs, 'Deep Learning Model');
        
        this.updateStatus(`Recommendations generated for user ${userId}`);
    }

    createComparisonTable(userTopRated, recommendations, modelName) {
        let html = `
            <table>
                <thead>
                    <tr>
                        <th>User's Top Rated Movies</th>
                        <th>${modelName} Recommendations</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        for (let i = 0; i < 10; i++) {
            const rated = userTopRated[i] || { title: '-', year: '', rating: '', genres: '' };
            const rec = recommendations[i] || { title: '-', year: '', score: '', genres: '' };
            
            html += `
                <tr>
                    <td>
                        <strong>${rated.title}</strong>${rated.year ? ` (${rated.year})` : ''}<br>
                        <small>Rating: ${rated.rating} | ${rated.genres}</small>
                    </td>
                    <td>
                        <strong>${rec.title}</strong>${rec.year ? ` (${rec.year})` : ''}<br>
                        <small>Score: ${rec.score} | ${rec.genres}</small>
                    </td>
                </tr>
            `;
        }
        
        html += `</tbody></table>`;
        return html;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
