// app.js
class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.users = new Map();
        this.userIndexMap = new Map();
        this.itemIndexMap = new Map();
        this.reverseUserIndex = new Map();
        this.reverseItemIndex = new Map();
        this.model = null;
        this.isDataLoaded = false;
        this.isModelTrained = false;
        
        this.initializeEventListeners();
        this.setupCharts();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.trainModel());
        document.getElementById('test').addEventListener('click', () => this.testModel());
    }

    setupCharts() {
        this.lossChart = this.createChart('lossChart', 'Training Loss', 'Epoch', 'Loss');
        this.embeddingChart = this.createChart('embeddingChart', 'Item Embeddings', 'X', 'Y');
    }

    createChart(canvasId, title, xLabel, yLabel) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        return {
            canvas,
            ctx,
            title,
            xLabel,
            yLabel,
            data: []
        };
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }

    async loadData() {
        this.updateStatus('Loading MovieLens data...');
        
        try {
            // Load interactions data
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            
            // Load items data
            const itemsResponse = await fetch('data/u.item');
            const itemsText = await itemsResponse.text();
            
            this.parseInteractions(interactionsText);
            this.parseItems(itemsText);
            this.buildIndexMaps();
            this.analyzeUserBehavior();
            
            this.isDataLoaded = true;
            document.getElementById('train').disabled = false;
            this.updateStatus(`Data loaded: ${this.interactions.length} interactions, ${this.items.size} movies, ${this.users.size} users`);
            
        } catch (error) {
            this.updateStatus('Error loading data: ' + error.message);
        }
    }

    parseInteractions(dataText) {
        const lines = dataText.split('\n');
        this.interactions = [];
        
        for (const line of lines) {
            if (line.trim()) {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                if (userId && itemId) {
                    const interaction = {
                        userId: parseInt(userId),
                        itemId: parseInt(itemId),
                        rating: parseFloat(rating),
                        timestamp: parseInt(timestamp)
                    };
                    this.interactions.push(interaction);
                    
                    // Build user data structure
                    if (!this.users.has(interaction.userId)) {
                        this.users.set(interaction.userId, {
                            id: interaction.userId,
                            ratings: [],
                            ratedItems: new Set()
                        });
                    }
                    const user = this.users.get(interaction.userId);
                    user.ratings.push(interaction);
                    user.ratedItems.add(interaction.itemId);
                }
            }
        }
    }

    parseItems(dataText) {
        const lines = dataText.split('\n');
        this.items.clear();
        
        for (const line of lines) {
            if (line.trim()) {
                const parts = line.split('|');
                if (parts.length >= 24) {
                    const itemId = parseInt(parts[0]);
                    const title = parts[1];
                    
                    // Extract year from title
                    const yearMatch = title.match(/\((\d{4})\)/);
                    const year = yearMatch ? parseInt(yearMatch[1]) : null;
                    
                    // Parse genre features (last 19 fields)
                    const genreFeatures = parts.slice(5, 24).map(x => parseInt(x));
                    
                    this.items.set(itemId, {
                        id: itemId,
                        title: title.replace(/\(\d{4}\)/, '').trim(),
                        year: year,
                        genres: genreFeatures
                    });
                }
            }
        }
    }

    buildIndexMaps() {
        // Create 0-based indices for users and items
        const uniqueUsers = Array.from(this.users.keys()).sort((a, b) => a - b);
        const uniqueItems = Array.from(this.items.keys()).sort((a, b) => a - b);
        
        this.userIndexMap.clear();
        this.itemIndexMap.clear();
        this.reverseUserIndex.clear();
        this.reverseItemIndex.clear();
        
        uniqueUsers.forEach((userId, index) => {
            this.userIndexMap.set(userId, index);
            this.reverseUserIndex.set(index, userId);
        });
        
        uniqueItems.forEach((itemId, index) => {
            this.itemIndexMap.set(itemId, index);
            this.reverseItemIndex.set(index, itemId);
        });
    }

    analyzeUserBehavior() {
        // Calculate average ratings and sort user ratings
        for (const user of this.users.values()) {
            user.ratings.sort((a, b) => {
                if (b.rating !== a.rating) {
                    return b.rating - a.rating;
                }
                return b.timestamp - a.timestamp;
            });
            
            user.averageRating = user.ratings.reduce((sum, r) => sum + r.rating, 0) / user.ratings.length;
        }
    }

    async trainModel() {
        if (!this.isDataLoaded) {
            this.updateStatus('Please load data first');
            return;
        }
        
        this.updateStatus('Initializing model...');
        
        const config = {
            numUsers: this.users.size,
            numItems: this.items.size,
            embeddingDim: 32,
            learningRate: 0.001,
            epochs: 50,
            batchSize: 1024,
            useMLP: true
        };
        
        this.model = new TwoTowerModel(config);
        
        this.updateStatus('Starting training...');
        document.getElementById('train').disabled = true;
        
        try {
            const lossHistory = await this.model.train(
                this.interactions,
                this.userIndexMap,
                this.itemIndexMap,
                (epoch, loss) => this.onTrainingProgress(epoch, loss)
            );
            
            this.isModelTrained = true;
            document.getElementById('test').disabled = false;
            this.updateStatus(`Training completed! Final loss: ${lossHistory[lossHistory.length - 1].toFixed(4)}`);
            
            // Visualize embeddings
            await this.visualizeEmbeddings();
            
        } catch (error) {
            this.updateStatus('Training error: ' + error.message);
            document.getElementById('train').disabled = false;
        }
    }

    onTrainingProgress(epoch, loss) {
        this.updateStatus(`Epoch ${epoch + 1}: loss = ${loss.toFixed(4)}`);
        
        // Update loss chart
        this.lossChart.data.push({ x: epoch + 1, y: loss });
        this.drawLineChart(this.lossChart);
    }

    drawLineChart(chart) {
        const { ctx, canvas, data, title, xLabel, yLabel } = chart;
        
        if (data.length === 0) return;
        
        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Set up margins and scaling
        const margin = 40;
        const width = canvas.width - 2 * margin;
        const height = canvas.height - 2 * margin;
        
        // Find data ranges
        const xValues = data.map(d => d.x);
        const yValues = data.map(d => d.y);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        // Scale functions
        const scaleX = (x) => margin + (x - xMin) / (xMax - xMin) * width;
        const scaleY = (y) => canvas.height - margin - (y - yMin) / (yMax - yMin) * height;
        
        // Draw axes
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, canvas.height - margin);
        ctx.lineTo(canvas.width - margin, canvas.height - margin);
        ctx.stroke();
        
        // Draw labels
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(xLabel, canvas.width / 2, canvas.height - 10);
        ctx.save();
        ctx.translate(10, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
        
        ctx.textAlign = 'left';
        ctx.fillText(title, margin, 20);
        
        // Draw data line
        if (data.length > 1) {
            ctx.strokeStyle = '#007cba';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(scaleX(data[0].x), scaleY(data[0].y));
            
            for (let i = 1; i < data.length; i++) {
                ctx.lineTo(scaleX(data[i].x), scaleY(data[i].y));
            }
            ctx.stroke();
        }
        
        // Draw data points
        ctx.fillStyle = '#007cba';
        for (const point of data) {
            ctx.beginPath();
            ctx.arc(scaleX(point.x), scaleY(point.y), 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    async visualizeEmbeddings() {
        if (!this.model) return;
        
        this.updateStatus('Computing embedding visualization...');
        
        try {
            // Get a sample of item embeddings
            const sampleSize = Math.min(200, this.items.size);
            const sampleIndices = [];
            const sampleTitles = [];
            
            // Sample diverse items
            const itemsArray = Array.from(this.items.values());
            for (let i = 0; i < sampleSize; i++) {
                const item = itemsArray[Math.floor(Math.random() * itemsArray.length)];
                const itemIndex = this.itemIndexMap.get(item.id);
                sampleIndices.push(itemIndex);
                sampleTitles.push(item.title);
            }
            
            const embeddings = await this.model.getItemEmbeddings(sampleIndices);
            const embeddingArray = await embeddings.array();
            
            // Simple PCA using power iteration
            const projected = this.simplePCA(embeddingArray, 2);
            
            // Draw on canvas
            this.drawEmbeddingProjection(projected, sampleTitles);
            
        } catch (error) {
            console.error('Error visualizing embeddings:', error);
        }
    }

    simplePCA(embeddings, dimensions = 2) {
        // Center the data
        const centered = [];
        const means = [];
        
        for (let j = 0; j < embeddings[0].length; j++) {
            let sum = 0;
            for (let i = 0; i < embeddings.length; i++) {
                sum += embeddings[i][j];
            }
            means[j] = sum / embeddings.length;
        }
        
        for (let i = 0; i < embeddings.length; i++) {
            centered.push(embeddings[i].map((val, j) => val - means[j]));
        }
        
        // Simple 2D projection using first two principal components
        const projected = [];
        for (let i = 0; i < centered.length; i++) {
            // Use first two dimensions as simple projection (for demo purposes)
            // In production, you'd implement proper PCA
            const x = centered[i][0] + centered[i][1] * 0.5;
            const y = centered[i][2] + centered[i][3] * 0.5;
            projected.push({ x, y });
        }
        
        return projected;
    }

    drawEmbeddingProjection(projections, titles) {
        const { ctx, canvas } = this.embeddingChart;
        
        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Find bounds
        const xValues = projections.map(p => p.x);
        const yValues = projections.map(p => p.y);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        // Scale function
        const scaleX = (x) => (x - xMin) / (xMax - xMin) * (canvas.width - 40) + 20;
        const scaleY = (y) => (y - yMin) / (yMax - yMin) * (canvas.height - 40) + 20;
        
        // Draw points
        ctx.fillStyle = '#007cba';
        for (let i = 0; i < projections.length; i++) {
            const x = scaleX(projections[i].x);
            const y = scaleY(projections[i].y);
            
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // Add title
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Item Embeddings (2D Projection)', canvas.width / 2, 20);
    }

    async testModel() {
        if (!this.isModelTrained) {
            this.updateStatus('Please train the model first');
            return;
        }
        
        this.updateStatus('Generating recommendations...');
        
        try {
            // Find users with at least 20 ratings
            const qualifiedUsers = Array.from(this.users.values())
                .filter(user => user.ratings.length >= 20);
            
            if (qualifiedUsers.length === 0) {
                this.updateStatus('No users with sufficient ratings found');
                return;
            }
            
            // Pick random user
            const randomUser = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
            const userIndex = this.userIndexMap.get(randomUser.id);
            
            // Get user's top 10 rated movies
            const topRated = randomUser.ratings.slice(0, 10).map(rating => ({
                title: this.items.get(rating.itemId).title,
                rating: rating.rating
            }));
            
            // Get model recommendations
            const recommendations = await this.model.getRecommendations(
                userIndex,
                randomUser.ratedItems,
                this.itemIndexMap,
                this.reverseItemIndex,
                this.items,
                10
            );
            
            // Display results
            this.displayRecommendations(randomUser, topRated, recommendations);
            this.updateStatus(`Recommendations generated for user ${randomUser.id}`);
            
        } catch (error) {
            this.updateStatus('Error generating recommendations: ' + error.message);
        }
    }

    displayRecommendations(user, topRated, recommendations) {
        // Update user info
        document.getElementById('userInfo').innerHTML = `
            <p><strong>User ID:</strong> ${user.id} | 
            <strong>Total Ratings:</strong> ${user.ratings.length} | 
            <strong>Average Rating:</strong> ${user.averageRating.toFixed(2)}</p>
        `;
        
        // Populate top rated table
        const ratingsTable = document.getElementById('userRatingsTable').querySelector('tbody');
        ratingsTable.innerHTML = '';
        
        topRated.forEach(movie => {
            const row = ratingsTable.insertRow();
            row.insertCell(0).textContent = movie.title;
            row.insertCell(1).textContent = '★'.repeat(Math.round(movie.rating)) + ` (${movie.rating})`;
        });
        
        // Populate recommendations table
        const recsTable = document.getElementById('recommendationsTable').querySelector('tbody');
        recsTable.innerHTML = '';
        
        recommendations.forEach(rec => {
            const row = recsTable.insertRow();
            row.insertCell(0).textContent = rec.title;
            row.insertCell(1).textContent = rec.score.toFixed(4);
        });
    }

    // Non-deep learning method for comparison
    getNonDLRecommendations(userId, topK = 10) {
        const user = this.users.get(userId);
        if (!user) return [];
        
        // Create user profile as average of rated movie genre vectors
        const userProfile = new Array(19).fill(0);
        let ratedCount = 0;
        
        for (const rating of user.ratings) {
            const item = this.items.get(rating.itemId);
            if (item && item.genres) {
                for (let i = 0; i < 19; i++) {
                    userProfile[i] += item.genres[i] * rating.rating;
                }
                ratedCount++;
            }
        }
        
        // Normalize
        if (ratedCount > 0) {
            for (let i = 0; i < 19; i++) {
                userProfile[i] /= ratedCount;
            }
        }
        
        // Calculate cosine similarity with all unrated items
        const scores = [];
        const userNorm = Math.sqrt(userProfile.reduce((sum, val) => sum + val * val, 0));
        
        for (const [itemId, item] of this.items) {
            if (!user.ratedItems.has(itemId)) {
                const itemVector = item.genres;
                const itemNorm = Math.sqrt(itemVector.reduce((sum, val) => sum + val * val, 0));
                
                if (userNorm > 0 && itemNorm > 0) {
                    let dotProduct = 0;
                    for (let i = 0; i < 19; i++) {
                        dotProduct += userProfile[i] * itemVector[i];
                    }
                    const similarity = dotProduct / (userNorm * itemNorm);
                    scores.push({
                        itemId,
                        title: item.title,
                        score: similarity
                    });
                }
            }
        }
        
        // Return top K
        return scores.sort((a, b) => b.score - a.score).slice(0, topK);
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
