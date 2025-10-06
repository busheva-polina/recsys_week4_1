// two-tower.js
class TraditionalTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        
        // Initialize embeddings with random values
        this.userEmbeddings = Array.from({length: numUsers}, () => 
            Array.from({length: embeddingDim}, () => (Math.random() - 0.5) * 0.1)
        );
        this.itemEmbeddings = Array.from({length: numItems}, () => 
            Array.from({length: embeddingDim}, () => (Math.random() - 0.5) * 0.1)
        );
        
        this.learningRate = 0.01;
    }

    async train(userIndices, itemIndices, progressCallback) {
        const epochs = 50;
        const batchSize = 512;
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            let batchCount = 0;
            
            // Process in batches
            for (let i = 0; i < userIndices.length; i += batchSize) {
                const batchUsers = userIndices.slice(i, i + batchSize);
                const batchItems = itemIndices.slice(i, i + batchSize);
                
                const batchLoss = this.trainBatch(batchUsers, batchItems);
                totalLoss += batchLoss;
                batchCount++;
            }
            
            const avgLoss = totalLoss / batchCount;
            if (progressCallback) {
                progressCallback(avgLoss, epoch + 1);
            }
            
            // Early stopping if loss is NaN
            if (isNaN(avgLoss)) break;
        }
    }

    trainBatch(userIndices, itemIndices) {
        let totalLoss = 0;
        
        for (let i = 0; i < userIndices.length; i++) {
            const userIdx = userIndices[i];
            const positiveItemIdx = itemIndices[i];
            
            // Get embeddings
            const userEmb = this.userEmbeddings[userIdx];
            const positiveEmb = this.itemEmbeddings[positiveItemIdx];
            
            // Calculate positive score
            const positiveScore = this.dotProduct(userEmb, positiveEmb);
            
            // Sample negative item
            let negativeItemIdx;
            do {
                negativeItemIdx = Math.floor(Math.random() * this.numItems);
            } while (negativeItemIdx === positiveItemIdx);
            
            const negativeEmb = this.itemEmbeddings[negativeItemIdx];
            const negativeScore = this.dotProduct(userEmb, negativeEmb);
            
            // BPR loss: -log(sigmoid(positiveScore - negativeScore))
            const diff = positiveScore - negativeScore;
            const sigmoid = 1 / (1 + Math.exp(-diff));
            const loss = -Math.log(sigmoid + 1e-8);
            
            totalLoss += loss;
            
            // Calculate gradients
            const gradFactor = (1 - sigmoid);
            
            // Update embeddings
            for (let d = 0; d < this.embeddingDim; d++) {
                // User gradient
                this.userEmbeddings[userIdx][d] += this.learningRate * gradFactor * 
                    (positiveEmb[d] - negativeEmb[d]);
                
                // Positive item gradient
                this.itemEmbeddings[positiveItemIdx][d] += this.learningRate * gradFactor * userEmb[d];
                
                // Negative item gradient
                this.itemEmbeddings[negativeItemIdx][d] -= this.learningRate * gradFactor * userEmb[d];
            }
        }
        
        return totalLoss / userIndices.length;
    }

    dotProduct(a, b) {
        let result = 0;
        for (let i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    getItemEmbeddings() {
        return this.itemEmbeddings;
    }

    async getRecommendations(userIndex, allItemIndices, excludeIndices) {
        const userEmb = this.userEmbeddings[userIndex];
        const excludeSet = new Set(excludeIndices);
        
        // Calculate scores for all items
        const scores = [];
        for (const itemIdx of allItemIndices) {
            if (!excludeSet.has(itemIdx)) {
                const itemEmb = this.itemEmbeddings[itemIdx];
                const score = this.dotProduct(userEmb, itemEmb);
                scores.push({index: itemIdx, score: score});
            }
        }
        
        // Sort by score descending and return top 10
        return scores.sort((a, b) => b.score - a.score)
                    .slice(0, 10)
                    .map(item => item.index);
    }
}

class DeepLearningTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, numGenres) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.numGenres = numGenres;
        
        // User tower: user_id -> embedding
        this.userEmbeddings = tf.variable(tf.randomNormal([numUsers, embeddingDim], 0, 0.05));
        
        // Item tower: item_id -> embedding + genre features -> MLP
        this.itemEmbeddings = tf.variable(tf.randomNormal([numItems, embeddingDim], 0, 0.05));
        
        // MLP layers for item features
        this.itemHiddenWeights = tf.variable(tf.randomNormal([embeddingDim + numGenres, embeddingDim * 2], 0, 0.05));
        this.itemHiddenBias = tf.variable(tf.zeros([embeddingDim * 2]));
        this.itemOutputWeights = tf.variable(tf.randomNormal([embeddingDim * 2, embeddingDim], 0, 0.05));
        this.itemOutputBias = tf.variable(tf.zeros([embeddingDim]));
        
        this.optimizer = tf.train.adam(0.001);
    }

    async train(userIndices, itemIndices, itemGenreFeatures, progressCallback) {
        const epochs = 50;
        const batchSize = 512;
        
        // Convert genre features to tensor
        this.itemGenreTensor = tf.tensor2d(itemGenreFeatures);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            let batchCount = 0;
            
            // Shuffle data
            const shuffledIndices = this.shuffleArray([...userIndices.keys()]);
            
            for (let i = 0; i < shuffledIndices.length; i += batchSize) {
                const batchIndices = shuffledIndices.slice(i, i + batchSize);
                const batchUsers = batchIndices.map(idx => userIndices[idx]);
                const batchItems = batchIndices.map(idx => itemIndices[idx]);
                
                const loss = await this.trainBatch(batchUsers, batchItems);
                totalLoss += loss;
                batchCount++;
            }
            
            const avgLoss = totalLoss / batchCount;
            if (progressCallback) {
                progressCallback(avgLoss, epoch + 1);
            }
            
            if (isNaN(avgLoss)) break;
        }
    }

    async trainBatch(userIndices, itemIndices) {
        return tf.tidy(() => {
            const users = tf.tensor1d(userIndices, 'int32');
            const items = tf.tensor1d(itemIndices, 'int32');
            
            const lossFunction = () => {
                // Get user embeddings
                const userEmb = this.userEmbeddings.gather(users);
                
                // Get item embeddings and process through MLP
                const itemBaseEmb = this.itemEmbeddings.gather(items);
                const itemGenreFeatures = this.itemGenreTensor.gather(items);
                const itemCombined = tf.concat([itemBaseEmb, itemGenreFeatures], 1);
                
                // MLP with one hidden layer and ReLU activation
                const hidden = tf.relu(tf.add(tf.matMul(itemCombined, this.itemHiddenWeights), this.itemHiddenBias));
                const itemEmb = tf.add(tf.matMul(hidden, this.itemOutputWeights), this.itemOutputBias);
                
                // Normalize embeddings
                const userEmbNorm = tf.div(userEmb, tf.norm(userEmb, 2, 1, true));
                const itemEmbNorm = tf.div(itemEmb, tf.norm(itemEmb, 2, 1, true));
                
                // Calculate scores using dot product
                const scores = tf.sum(tf.mul(userEmbNorm, itemEmbNorm), 1);
                
                // Sample negative items
                const negativeItems = tf.randomUniform([userIndices.length], 0, this.numItems, 'int32');
                const negativeBaseEmb = this.itemEmbeddings.gather(negativeItems);
                const negativeGenreFeatures = this.itemGenreTensor.gather(negativeItems);
                const negativeCombined = tf.concat([negativeBaseEmb, negativeGenreFeatures], 1);
                
                const negativeHidden = tf.relu(tf.add(tf.matMul(negativeCombined, this.itemHiddenWeights), this.itemHiddenBias));
                const negativeEmb = tf.add(tf.matMul(negativeHidden, this.itemOutputWeights), this.itemOutputBias);
                const negativeEmbNorm = tf.div(negativeEmb, tf.norm(negativeEmb, 2, 1, true));
                
                const negativeScores = tf.sum(tf.mul(userEmbNorm, negativeEmbNorm), 1);
                
                // BPR loss
                const diff = tf.sub(scores, negativeScores);
                const loss = tf.mean(tf.neg(tf.log(tf.sigmoid(diff))));
                
                return loss;
            };
            
            const loss = this.optimizer.minimize(lossFunction, true);
            return loss ? loss.dataSync()[0] : 0;
        });
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    getItemEmbeddings() {
        return tf.tidy(() => {
            // Get all item embeddings processed through MLP
            const itemBaseEmb = this.itemEmbeddings;
            const itemGenreFeatures = this.itemGenreTensor;
            const itemCombined = tf.concat([itemBaseEmb, itemGenreFeatures], 1);
            
            const hidden = tf.relu(tf.add(tf.matMul(itemCombined, this.itemHiddenWeights), this.itemHiddenBias));
            const itemEmb = tf.add(tf.matMul(hidden, this.itemOutputWeights), this.itemOutputBias);
            
            return itemEmb.arraySync();
        });
    }

    async getRecommendations(userIndex, allItemIndices, excludeIndices) {
        return tf.tidy(() => {
            const userEmb = this.userEmbeddings.gather([userIndex]);
            const userEmbNorm = tf.div(userEmb, tf.norm(userEmb, 2, 1, true));
            
            // Get all item embeddings
            const items = tf.tensor1d(allItemIndices, 'int32');
            const itemBaseEmb = this.itemEmbeddings.gather(items);
            const itemGenreFeatures = this.itemGenreTensor.gather(items);
            const itemCombined = tf.concat([itemBaseEmb, itemGenreFeatures], 1);
            
            const hidden = tf.relu(tf.add(tf.matMul(itemCombined, this.itemHiddenWeights), this.itemHiddenBias));
            const itemEmb = tf.add(tf.matMul(hidden, this.itemOutputWeights), this.itemOutputBias);
            const itemEmbNorm = tf.div(itemEmb, tf.norm(itemEmb, 2, 1, true));
            
            // Calculate scores
            const scores = tf.matMul(userEmbNorm, itemEmbNorm.transpose());
            
            // Apply mask for excluded items
            const excludeSet = new Set(excludeIndices);
            const mask = allItemIndices.map((idx, i) => excludeSet.has(idx) ? -Infinity : 0);
            const maskTensor = tf.tensor1d(mask);
            const maskedScores = tf.add(scores.flatten(), maskTensor);
            
           
