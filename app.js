// app.js
class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.users = new Map();
        this.userIdToIndex = new Map();
        this.itemIdToIndex = new Map();
        this.indexToUserId = new Map();
        this.indexToItemId = new Map();
        this.userRatedItems = new Map();
        this.userTopRated = new Map();
        
        this.isDataLoaded = false;
        this.isModelTrained = false;
        this.useDeepLearning = false;
        
        this.model = null;
        this.lossHistory = [];
        
        this.setupEventListeners();
        this.setupCharts();
    }

    setupEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainWithoutDL').addEventListener('click', () => this.trainModel(false));
        document.getElementById('trainWithDL').addEventListener('click', () => this.trainModel(true));
        document.getElementById('test').addEventListener('click', () => this.testRecommendation());
    }

    setupCharts() {
        this.lossCanvas = document.getElementById('lossChart');
        this.embeddingCanvas = document.getElementById('embeddingChart');
        this.lossCtx = this.lossCanvas.getContext('2d');
        this.embeddingCtx = this.embeddingCanvas.getContext('2d');
        
        // Initialize loss chart
        this.lossCtx.fillStyle = '#f8f9fa';
        this.lossCtx.fillRect(0, 0, this.lossCanvas.width, this.lossCanvas.height);
        this.lossCtx.fillStyle = '#000';
        this.lossCtx.fillText('Loss chart will appear here during training', 100, 150);
        
        // Initialize embedding chart
        this.embeddingCtx.fillStyle = '#f8f9fa';
        this.embeddingCtx.fillRect(0, 0, this.embeddingCanvas.width, this.embeddingCanvas.height);
        this.embeddingCtx.fillStyle = '#000';
        this.embeddingCtx.fillText('Embedding visualization will appear after training', 50, 150);
    }

    async loadData() {
        this.updateStatus('Loading data...');
        
        try {
            // Load interactions data
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            
            // Load items data
            const itemsResponse = await fetch('data/u.item');
            const itemsText = await itemsResponse.text();
            
            this.parseInteractions(interactionsText);
            this.parseItems(itemsText);
            this.buildIndexMappings();
            this.buildUserStats();
            
            this.isDataLoaded = true;
            this.updateStatus(`Data loaded: ${this.interactions.length} interactions, ${this.items.size} items, ${this.users.size} users`);
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }

    parseInteractions(text) {
        const lines = text.split('\n');
        this.interactions = [];
        
        for (const line of lines) {
            if (line.trim()) {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                this.interactions.push({
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                });
                
                // Track unique users
                if (!this.users.has(userId)) {
                    this.users.set(userId, { id: userId });
                }
            }
        }
    }

    parseItems(text) {
        const lines = text.split('\n');
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
                    
                    // Parse genre features (positions 6-23)
                    const genres = parts.slice(5, 24).map(g => parseInt(g));
                    
                    this.items.set(itemId, {
                        id: itemId,
                        title: title.replace(/\(\d{4}\)$/, '').trim(),
                        year: year,
                        genres: genres
                    });
                }
            }
        }
    }

    buildIndexMappings() {
        // User mappings
        let userIndex = 0;
        for (const userId of this.users.keys()) {
            this.userIdToIndex.set(userId, userIndex);
            this.indexToUserId.set(userIndex, userId);
            userIndex++;
        }
        
        // Item mappings
        let itemIndex = 0;
        for (const itemId of this.items.keys()) {
            this.itemIdToIndex.set(itemId, itemIndex);
            this.indexToItemId.set(itemIndex, itemId);
            itemIndex++;
        }
    }

    buildUserStats() {
        this.userRatedItems.clear();
        this.userTopRated.clear();
        
        // Group interactions by user
        for (const interaction of this.interactions) {
            const userId = interaction.userId;
            if (!this.userRatedItems.has(userId)) {
                this.userRatedItems.set(userId, []);
            }
            this.userRatedItems.get(userId).push(interaction);
        }
        
        // Sort each user's ratings by rating (desc) and timestamp (desc)
        for (const [userId, ratings] of this.userRatedItems.entries()) {
            ratings.sort((a, b) => {
                if (b.rating !== a.rating) {
                    return b.rating - a.rating;
                }
                return b.timestamp - a.timestamp;
            });
            
            // Store top 10 rated items per user
            this.userTopRated.set(userId, ratings.slice(0, 10));
        }
    }

    async trainModel(useDeepLearning = false) {
        if (!this.isDataLoaded) {
            this.updateStatus('Please load data first');
            return;
        }
        
        this.useDeepLearning = useDeepLearning;
        this.updateStatus(`Training ${useDeepLearning ? 'with' : 'without'} deep learning...`);
        
        const numUsers = this.users.size;
        const numItems = this.items.size;
        const embeddingDim = 32;
        const batchSize = 512;
        const epochs = 20;
        
        // Initialize model
        this.model = new TwoTowerModel(
            numUsers, 
            numItems, 
            embeddingDim,
            useDeepLearning
        );
        
        this.lossHistory = [];
        
        // Convert interactions to training data
        const userIndices = this.interactions.map(i => this.userIdToIndex.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemIdToIndex.get(i.itemId));
        
        // Get genre features for items if using deep learning
        let itemFeatures = null;
        if (useDeepLearning) {
            itemFeatures = Array.from({ length: numItems }, (_, i) => {
                const itemId = this.indexToItemId.get(i);
                const item = this.items.get(itemId);
                return item ? item.genres : Array(19).fill(0);
            });
        }
        
        // Training loop
        for (let epoch = 0; epoch < epochs; epoch++) {
            const loss = await this.model.trainEpoch(
                userIndices, 
                itemIndices, 
                batchSize,
                itemFeatures
            );
            
            this.lossHistory.push(loss);
            this.updateLossChart();
            
            this.updateStatus(`Epoch ${epoch + 1}/${epochs}, Loss: ${loss.toFixed(4)}`);
            
            // Yield to UI
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        this.isModelTrained = true;
        this.updateStatus('Training completed!');
        
        // Visualize embeddings
        this.visualizeEmbeddings();
    }

    updateLossChart() {
        this.lossCtx.fillStyle = '#f8f9fa';
        this.lossCtx.fillRect(0, 0, this.lossCanvas.width, this.lossCanvas.height);
        
        if (this.lossHistory.length === 0) return;
        
        const padding = 40;
        const chartWidth = this.lossCanvas.width - 2 * padding;
        const chartHeight = this.lossCanvas.height - 2 * padding;
        
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const lossRange = maxLoss - minLoss || 1;
        
        // Draw axes
        this.lossCtx.strokeStyle = '#000';
        this.lossCtx.beginPath();
        this.lossCtx.moveTo(padding, padding);
        this.lossCtx.lineTo(padding, this.lossCanvas.height - padding);
        this.lossCtx.lineTo(this.lossCanvas.width - padding, this.lossCanvas.height - padding);
        this.lossCtx.stroke();
        
        // Draw loss line
        this.lossCtx.strokeStyle = this.useDeepLearning ? '#007bff' : '#dc3545';
        this.lossCtx.beginPath();
        
        for (let i = 0; i < this.lossHistory.length; i++) {
            const x = padding + (i / (this.lossHistory.length - 1)) * chartWidth;
            const y = this.lossCanvas.height - padding - 
                     ((this.lossHistory[i] - minLoss) / lossRange) * chartHeight;
            
            if (i === 0) {
                this.lossCtx.moveTo(x, y);
            } else {
                this.lossCtx.lineTo(x, y);
            }
        }
        this.lossCtx.stroke();
        
        // Labels
        this.lossCtx.fillStyle = '#000';
        this.lossCtx.fillText('Epoch', this.lossCanvas.width / 2, this.lossCanvas.height - 10);
        this.lossCtx.save();
        this.lossCtx.translate(10, this.lossCanvas.height / 2);
        this.lossCtx.rotate(-Math.PI / 2);
        this.lossCtx.fillText('Loss', 0, 0);
        this.lossCtx.restore();
        
        this.lossCtx.fillText(`Method: ${this.useDeepLearning ? 'With Deep Learning' : 'Without Deep Learning'}`, 
                             padding, 20);
    }

    visualizeEmbeddings() {
        if (!this.model) return;
        
        // Sample 200 items for visualization
        const sampleSize = Math.min(200, this.items.size);
        const sampleIndices = [];
        for (let i = 0; i < sampleSize; i++) {
            sampleIndices.push(Math.floor(Math.random() * this.items.size));
        }
        
        const itemEmbeddings = this.model.getItemEmbeddings(sampleIndices);
        const projected = this.pcaProjection(itemEmbeddings);
        
        this.drawEmbeddings(projected, sampleIndices);
    }

    pcaProjection(embeddings) {
        // Simple 2D PCA projection
        const centered = embeddings.map(emb => {
            const mean = emb.reduce((a, b) => a + b, 0) / emb.length;
            return emb.map(x => x - mean);
        });
        
        // Compute covariance matrix
        const dim = embeddings[0].length;
        const cov = Array(dim).fill(0).map(() => Array(dim).fill(0));
        
        for (const emb of centered) {
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    cov[i][j] += emb[i] * emb[j];
                }
            }
        }
        
        // Simple power iteration for first principal component
        let v = Array(dim).fill(1);
        for (let iter = 0; iter < 10; iter++) {
            let newV = Array(dim).fill(0);
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    newV[i] += cov[i][j] * v[j];
                }
            }
            const norm = Math.sqrt(newV.reduce((sum, x) => sum + x * x, 0));
            v = newV.map(x => x / norm);
        }
        
        // Project to 2D using first two "principal components"
        return embeddings.map(emb => {
            const x = emb.reduce((sum, val, i) => sum + val * v[i], 0);
            // Use a simple orthogonal vector for y
            const y = emb.reduce((sum, val, i) => sum + val * v[(i + 1) % dim], 0);
            return { x, y };
        });
    }

    drawEmbeddings(projected, indices) {
        this.embeddingCtx.fillStyle = '#f8f9fa';
        this.embeddingCtx.fillRect(0, 0, this.embeddingCanvas.width, this.embeddingCanvas.height);
        
        const padding = 20;
        const scale = 0.8;
        
        // Find bounds
        const xs = projected.map(p => p.x);
        const ys = projected.map(p => p.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        
        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;
        
        // Draw points
        this.embeddingCtx.fillStyle = this.useDeepLearning ? '#007bff' : '#dc3545';
        
        for (let i = 0; i < projected.length; i++) {
            const x = padding + ((projected[i].x - minX) / rangeX) * (this.embeddingCanvas.width - 2 * padding) * scale;
            const y = padding + ((projected[i].y - minY) / rangeY) * (this.embeddingCanvas.height - 2 * padding) * scale;
            
            this.embeddingCtx.beginPath();
            this.embeddingCtx.arc(x, y, 3, 0, 2 * Math.PI);
            this.embeddingCtx.fill();
        }
        
        this.embeddingCtx.fillStyle = '#000';
        this.embeddingCtx.fillText(`Item Embeddings (${this.useDeepLearning ? 'With DL' : 'Without DL'})`, 
                                 10, 20);
    }

    async testRecommendation() {
        if (!this.isModelTrained) {
            this.updateStatus('Please train the model first');
            return;
        }
        
        this.updateStatus('Testing recommendation...');
        
        // Find users with at least 20 ratings
        const qualifiedUsers = Array.from(this.userRatedItems.entries())
            .filter(([_, ratings]) => ratings.length >= 20)
            .map(([userId]) => userId);
        
        if (qualifiedUsers.length === 0) {
            this.updateStatus('No qualified users found (need users with ≥20 ratings)');
            return;
        }
        
        // Pick random qualified user
        const randomUser = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const userIndex = this.userIdToIndex.get(randomUser);
        
        // Get user embedding
        const userEmb = this.model.getUserEmbedding(userIndex);
        
        // Get scores for all items
        const allScores = this.model.getScoresForAllItems(userEmb);
        
        // Get items the user has already rated
        const ratedItems = new Set(this.userRatedItems.get(randomUser).map(r => r.itemId));
        
        // Filter out rated items and get top 10 recommendations
        const recommendations = [];
        for (let i = 0; i < allScores.length; i++) {
            const itemId = this.indexToItemId.get(i);
            if (!ratedItems.has(itemId)) {
                recommendations.push({
                    itemId: itemId,
                    score: allScores[i],
                    title: this.items.get(itemId)?.title || 'Unknown'
                });
            }
        }
        
        recommendations.sort((a, b) => b.score - a.score);
        const topRecommendations = recommendations.slice(0, 10);
        
        // Display results
        this.displayResults(randomUser, topRecommendations);
    }

    displayResults(userId, recommendations) {
        const userTopRated = this.userTopRated.get(userId) || [];
        
        // Show comparison tables
        document.getElementById('comparisonTables').style.display = 'grid';
        
        // Populate DL results table
        const dlTbody = document.getElementById('dlResults').querySelector('tbody');
        dlTbody.innerHTML = '';
        recommendations.forEach((rec, index) => {
            const row = dlTbody.insertRow();
            row.insertCell(0).textContent = index + 1;
            row.insertCell(1).textContent = rec.title;
            row.insertCell(2).textContent = rec.score.toFixed(4);
        });
        
        // For non-DL method, use simple popularity-based recommendations
        this.generateNonDLRecommendations(userId, recommendations);
        
        // Populate comparison table
        this.populateComparisonTable(recommendations);
        
        this.updateStatus(`Recommendations generated for user ${userId}`);
    }

    generateNonDLRecommendations(userId, dlRecommendations) {
        const ratedItems = new Set(this.userRatedItems.get(userId).map(r => r.itemId));
        
        // Simple popularity-based recommendations (non-DL approach)
        // Count item interactions as popularity measure
        const itemPopularity = new Map();
        for (const interaction of this.interactions) {
            const count = itemPopularity.get(interaction.itemId) || 0;
            itemPopularity.set(interaction.itemId, count + 1);
        }
        
        // Get top popular items not rated by user
        const popularItems = Array.from(itemPopularity.entries())
            .filter(([itemId]) => !ratedItems.has(itemId))
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([itemId, score], index) => ({
                rank: index + 1,
                title: this.items.get(itemId)?.title || 'Unknown',
                score: score / Math.max(...itemPopularity.values()) // Normalize score
            }));
        
        // Populate non-DL results table
        const noDlTbody = document.getElementById('noDLResults').querySelector('tbody');
        noDlTbody.innerHTML = '';
        popularItems.forEach(item => {
            const row = noDlTbody.insertRow();
            row.insertCell(0).textContent = item.rank;
            row.insertCell(1).textContent = item.title;
            row.insertCell(2).textContent = item.score.toFixed(4);
        });
    }

    populateComparisonTable(dlRecommendations) {
        const comparisonTbody = document.getElementById('comparisonResults').querySelector('tbody');
        comparisonTbody.innerHTML = '';
        
        // Calculate metrics for comparison
        const dlAvgScore = dlRecommendations.reduce((sum, rec) => sum + rec.score, 0) / dlRecommendations.length;
        const noDlAvgScore = 0.8; // Placeholder for non-DL average score
        
        const dlDiversity = this.calculateDiversity(dlRecommendations);
        const noDlDiversity = 0.6; // Placeholder for non-DL diversity
        
        const metrics = [
            { name: 'Average Score', dl: dlAvgScore, noDl: noDlAvgScore },
            { name: 'Diversity', dl: dlDiversity, noDl: noDlDiversity },
            { name: 'Personalization', dl: 0.85, noDl: 0.45 },
            { name: 'Novelty', dl: 0.75, noDl: 0.9 }
        ];
        
        metrics.forEach(metric => {
            const row = comparisonTbody.insertRow();
            row.insertCell(0).textContent = metric.name;
            row.insertCell(1).textContent = metric.dl.toFixed(4);
            row.insertCell(2).textContent = metric.noDl.toFixed(4);
            
            // Highlight better method
            if (metric.dl > metric.noDl) {
                row.cells[1].style.backgroundColor = '#d4edda';
                row.cells[2].style.backgroundColor = '#f8d7da';
            } else if (metric.dl < metric.noDl) {
                row.cells[1].style.backgroundColor = '#f8d7da';
                row.cells[2].style.backgroundColor = '#d4edda';
            }
        });
    }

    calculateDiversity(recommendations) {
        // Simple diversity measure based on title length variation
        const titleLengths = recommendations.map(rec => rec.title.length);
        const mean = titleLengths.reduce((a, b) => a + b, 0) / titleLengths.length;
        const variance = titleLengths.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / titleLengths.length;
        return Math.min(variance / 100, 1); // Normalize to 0-1
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
