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
        
        this.traditionalModel = null;
        this.deepLearningModel = null;
        this.isDataLoaded = false;
        this.isTraditionalTrained = false;
        this.isDeepLearningTrained = false;
        
        this.traditionalLossData = [];
        this.deepLearningLossData = [];
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainTraditional').addEventListener('click', () => this.trainTraditional());
        document.getElementById('trainDeepLearning').addEventListener('click', () => this.trainDeepLearning());
        document.getElementById('test').addEventListener('click', () => this.testRecommendations());
    }

    async loadData() {
        this.updateStatus('Loading data...');
        
        try {
            // Load interactions data with error handling
            const interactionsResponse = await fetch('data/u.data');
            if (!interactionsResponse.ok) throw new Error('Failed to load u.data');
            const interactionsText = await interactionsResponse.text();
            
            // Load items data with error handling
            const itemsResponse = await fetch('data/u.item');
            if (!itemsResponse.ok) throw new Error('Failed to load u.item');
            const itemsText = await itemsResponse.text();
            
            this.parseInteractions(interactionsText);
            this.parseItems(itemsText);
            this.buildMappings();
            this.buildUserRatedItems();
            
            this.isDataLoaded = true;
            document.getElementById('trainTraditional').disabled = false;
            document.getElementById('trainDeepLearning').disabled = false;
            
            this.updateStatus(`Data loaded: ${this.interactions.length} interactions, ${this.items.size} items, ${this.userMap.size} users`);
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
            console.error('Data loading error:', error);
        }
    }

    parseInteractions(text) {
        const lines = text.split('\n').filter(line => line.trim());
        this.interactions = [];
        
        for (const line of lines) {
            // Handle both tab and space separation
            const parts = line.split(/\s+/).filter(part => part.trim());
            if (parts.length >= 3) {
                const [userId, itemId, rating, timestamp] = parts;
                this.interactions.push({
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: timestamp ? parseInt(timestamp) : Date.now()
                });
            }
        }
    }

    parseItems(text) {
        const lines = text.split('\n').filter(line => line.trim());
        this.items.clear();
        
        for (const line of lines) {
            // Handle pipe separation and fix parsing issues
            const parts = line.split('|').filter(part => part.trim());
            if (parts.length >= 2) {
                const itemId = parseInt(parts[0]);
                let title = parts[1];
                let year = null;
                
                // Extract year from title using safer regex
                const yearMatch = title.match(/\((\d{4})\)$/);
                if (yearMatch) {
                    year = parseInt(yearMatch[1]);
                    title = title.replace(/\(\d{4}\)$/, '').trim();
                }
                
                // Extract genres if available (positions 5-23 in original format)
                const genres = parts.length >= 24 ? parts.slice(5, 24) : [];
                
                this.items.set(itemId, {
                    title: title,
                    year: year,
                    genres: genres
                });
            }
        }
    }

    buildMappings() {
        // Build user and item mappings
        const uniqueUsers = [...new Set(this.interactions.map(i => i.userId))].sort((a, b) => a - b);
        const uniqueItems = [...new Set(this.interactions.map(i => i.itemId))].sort((a, b) => a - b);
        
        this.userMap.clear();
        this.itemMap.clear();
        this.reverseUserMap.clear();
        this.reverseItemMap.clear();
        
        uniqueUsers.forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });
        
        uniqueItems.forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
    }

    buildUserRatedItems() {
        this.userRatedItems.clear();
        this.userTopRated.clear();
        
        // Group interactions by user
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!this.userRatedItems.has(userId)) {
                this.userRatedItems.set(userId, []);
            }
            this.userRatedItems.get(userId).push(interaction);
        });
        
        // Sort each user's ratings by rating (desc) and timestamp (desc)
        this.userRatedItems.forEach((ratings, userId) => {
            const sorted = ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
            this.userTopRated.set(userId, sorted.slice(0, 10));
        });
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }

    async trainTraditional() {
        if (!this.isDataLoaded) return;
        
        this.updateStatus('Training traditional model...');
        document.getElementById('trainTraditional').disabled = true;
        
        this.traditionalModel = new TraditionalTwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            32
        );
        
        // Convert interactions to training format
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
        
        await this.traditionalModel.train(
            userIndices,
            itemIndices,
            (loss, epoch) => {
                this.traditionalLossData.push({x: epoch, y: loss});
                this.plotLoss('lossChartTraditional', this.traditionalLossData, 'Traditional Method Loss');
                this.updateStatus(`Training traditional model - Epoch ${epoch}, Loss: ${loss.toFixed(4)}`);
            }
        );
        
        this.isTraditionalTrained = true;
        this.updateTestButton();
        this.updateStatus('Traditional model training completed!');
        
        // Visualize embeddings
        this.visualizeEmbeddings(this.traditionalModel.getItemEmbeddings(), 'Traditional');
    }

    async trainDeepLearning() {
        if (!this.isDataLoaded) return;
        
        this.updateStatus('Training deep learning model...');
        document.getElementById('trainDeepLearning').disabled = true;
        
        // Prepare genre features
        const genreMap = new Map();
        let genreIndex = 0;
        
        this.items.forEach((item, itemId) => {
            if (item.genres) {
                item.genres.forEach(genre => {
                    if (genre === '1' && !genreMap.has(genreIndex)) {
                        // Map genre indices to feature names
                        const genreNames = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children', 
                                          'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                                          'Sci-Fi', 'Thriller', 'War', 'Western'];
                        const genreName = genreNames[genreIndex] || `Genre${genreIndex}`;
                        genreMap.set(genreIndex, genreName);
                    }
                    genreIndex++;
                });
            }
        });
        
        this.deepLearningModel = new DeepLearningTwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            32,
            genreMap.size
        );
        
        // Convert interactions to training format
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
        
        // Prepare genre features for each item
        const itemGenreFeatures = [];
        for (let i = 0; i < this.itemMap.size; i++) {
            const originalItemId = this.reverseItemMap.get(i);
            const item = this.items.get(originalItemId);
            const genreVector = new Array(genreMap.size).fill(0);
            
            if (item && item.genres) {
                item.genres.forEach((value, idx) => {
                    if (value === '1' && idx < genreMap.size) {
                        genreVector[idx] = 1;
                    }
                });
            }
            itemGenreFeatures.push(genreVector);
        }
        
        await this.deepLearningModel.train(
            userIndices,
            itemIndices,
            itemGenreFeatures,
            (loss, epoch) => {
                this.deepLearningLossData.push({x: epoch, y: loss});
                this.plotLoss('lossChartDeepLearning', this.deepLearningLossData, 'Deep Learning Loss');
                this.updateStatus(`Training deep learning model - Epoch ${epoch}, Loss: ${loss.toFixed(4)}`);
            }
        );
        
        this.isDeepLearningTrained = true;
        this.updateTestButton();
        this.updateStatus('Deep learning model training completed!');
        
        // Visualize embeddings
        this.visualizeEmbeddings(this.deepLearningModel.getItemEmbeddings(), 'Deep Learning');
    }

    updateTestButton() {
        document.getElementById('test').disabled = !(this.isTraditionalTrained || this.isDeepLearningTrained);
    }

    plotLoss(canvasId, data, title) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        if (data.length === 0) return;
        
        // Find data bounds
        const xValues = data.map(d => d.x);
        const yValues = data.map(d => d.y);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        // Add some padding
        const xRange = xMax - xMin || 1;
        const yRange = yMax - yMin || 1;
        const padding = 0.1;
        
        const scaleX = (x) => (x - xMin + padding * xRange) / (xRange * (1 + 2 * padding)) * width;
        const scaleY = (y) => height - (y - yMin + padding * yRange) / (yRange * (1 + 2 * padding)) * height;
        
        // Draw axes
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(scaleX(xMin), scaleY(0));
        ctx.lineTo(scaleX(xMax), scaleY(0));
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(scaleX(0), scaleY(yMin));
        ctx.lineTo(scaleX(0), scaleY(yMax));
        ctx.stroke();
        
        // Draw loss curve
        ctx.strokeStyle = '#007cba';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        data.forEach((point, i) => {
            if (i === 0) {
                ctx.moveTo(scaleX(point.x), scaleY(point.y));
            } else {
                ctx.lineTo(scaleX(point.x), scaleY(point.y));
            }
        });
        ctx.stroke();
        
        // Draw title
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText(title, 10, 20);
    }

    visualizeEmbeddings(embeddings, method) {
        if (!embeddings || embeddings.length === 0) return;
        
        // Sample embeddings for visualization (first 1000)
        const sampleSize = Math.min(1000, embeddings.length);
        const sampleEmbeddings = embeddings.slice(0, sampleSize);
        
        // Simple PCA implementation for 2D projection
        const projected = this.simplePCA(sampleEmbeddings, 2);
        
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Find bounds
        const xValues = projected.map(p => p[0]);
        const yValues = projected.map(p => p[1]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        const scaleX = (x) => (x - xMin) / (xMax - xMin) * (width - 40) + 20;
        const scaleY = (y) => (y - yMin) / (yMax - yMin) * (height - 40) + 20;
        
        // Draw points
        ctx.fillStyle = '#007cba';
        projected.forEach(point => {
            ctx.beginPath();
            ctx.arc(scaleX(point[0]), scaleY(point[1]), 2, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Draw title
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText(`${method} Item Embeddings (PCA)`, 10, 20);
    }

    simplePCA(embeddings, dimensions) {
        // Simple PCA using power iteration
        const result = [];
        
        for (let i = 0; i < embeddings.length; i++) {
            // For simplicity, just take first two dimensions of normalized embeddings
            const emb = embeddings[i];
            const norm = Math.sqrt(emb.reduce((sum, val) => sum + val * val, 0));
            const normalized = emb.map(val => val / norm);
            result.push([normalized[0] || 0, normalized[1] || 0]);
        }
        
        return result;
    }

    async testRecommendations() {
        if (!this.isTraditionalTrained && !this.isDeepLearningTrained) return;
        
        this.updateStatus('Testing recommendations...');
        
        // Find users with at least 20 ratings
        const qualifiedUsers = [...this.userRatedItems.entries()]
            .filter(([_, ratings]) => ratings.length >= 20)
            .map(([userId]) => userId);
        
        if (qualifiedUsers.length === 0) {
            this.updateStatus('No qualified users found (need users with ≥20 ratings)');
            return;
        }
        
        // Pick random user
        const randomUser = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const userIndex = this.userMap.get(randomUser);
        
        // Get user's top rated movies
        const topRated = this.userTopRated.get(randomUser) || [];
        
        // Generate recommendations from both models
        let traditionalRecs = [];
        let deepLearningRecs = [];
        
        if (this.isTraditionalTrained) {
            traditionalRecs = await this.traditionalModel.getRecommendations(
                userIndex, 
                Array.from(this.itemMap.values()), 
                topRated.map(r => this.itemMap.get(r.itemId))
            );
        }
        
        if (this.isDeepLearningTrained) {
            deepLearningRecs = await this.deepLearningModel.getRecommendations(
                userIndex,
                Array.from(this.itemMap.values()),
                topRated.map(r => this.itemMap.get(r.itemId))
            );
        }
        
        // Convert indices back to movie information
        const traditionalMovies = traditionalRecs.map(idx => {
            const originalId = this.reverseItemMap.get(idx);
            return this.items.get(originalId);
        }).filter(movie => movie);
        
        const deepLearningMovies = deepLearningRecs.map(idx => {
            const originalId = this.reverseItemMap.get(idx);
            return this.items.get(originalId);
        }).filter(movie => movie);
        
        // Display results
        this.displayResults(topRated, traditionalMovies, deepLearningMovies);
        this.updateStatus(`Recommendations generated for user ${randomUser}`);
    }

    displayResults(topRated, traditionalRecs, deepLearningRecs) {
        // Display traditional results
        const traditionalContainer = document.getElementById('resultsTraditional');
        traditionalContainer.innerHTML = this.createResultsTable(topRated, traditionalRecs, 'Traditional');
        
        // Display deep learning results
        const deepLearningContainer = document.getElementById('resultsDeepLearning');
        deepLearningContainer.innerHTML = this.createResultsTable(topRated, deepLearningRecs, 'Deep Learning');
        
        // Display comparison
        const comparisonContainer = document.getElementById('resultsComparison');
        comparisonContainer.innerHTML = this.createComparisonTable(traditionalRecs, deepLearningRecs);
    }

    createResultsTable(topRated, recommendations, method) {
        let html = `<table>
            <tr>
                <th>Top Rated Movies</th>
                <th>${method} Recommendations</th>
            </tr>`;
        
        const maxRows = Math.max(topRated.length, recommendations.length);
        
        for (let i = 0; i < maxRows; i++) {
            const rated = topRated[i];
            const rec = recommendations[i];
            
            html += '<tr>';
            
            // Top rated column
            if (rated) {
                const movie = this.items.get(rated.itemId);
                html += `<td>${movie ? movie.title : `Item ${rated.itemId}`} (${rated.rating}★)</td>`;
            } else {
                html += '<td></td>';
            }
            
            // Recommendations column
            if (rec) {
                html += `<td>${rec.title}${rec.year ? ` (${rec.year})` : ''}</td>`;
            } else {
                html += '<td></td>';
            }
            
            html += '</tr>';
        }
        
        html += '</table>';
        return html;
    }

    createComparisonTable(traditionalRecs, deepLearningRecs) {
        let html = `<table>
            <tr>
                <th>Traditional</th>
                <th>Deep Learning</th>
                <th>Match</th>
            </tr>`;
        
        const maxRows = Math.max(traditionalRecs.length, deepLearningRecs.length);
        const traditionalTitles = new Set(traditionalRecs.map(m => m.title));
        
        for (let i = 0; i < maxRows; i++) {
            const traditional = traditionalRecs[i];
            const deepLearning = deepLearningRecs[i];
            
            html += '<tr>';
            html += `<td>${traditional ? traditional.title : ''}</td>`;
            html += `<td>${deepLearning ? deepLearning.title : ''}</td>`;
            
            // Check if recommendations match
            const match = traditional && deepLearning && 
                         traditionalTitles.has(deepLearning.title) ? '✓' : '';
            html += `<td style="text-align: center;">${match}</td>`;
            html += '</tr>';
        }
        
        // Add summary row
        const overlap = traditionalRecs.filter(t => 
            deepLearningRecs.some(d => d && t && d.title === t.title)
        ).length;
        
        html += `<tr style="background-color: #f8f9fa;">
            <td colspan="2"><strong>Overlap: ${overlap} movies</strong></td>
            <td style="text-align: center;"><strong>${overlap}</strong></td>
        </tr>`;
        
        html += '</table>';
        return html;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
