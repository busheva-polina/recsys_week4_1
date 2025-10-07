// two-tower.js
class TwoTowerModel {
    constructor(numUsers, numItems, embeddingDim = 32, useDeepLearning = false) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.useDeepLearning = useDeepLearning;
        
        // Initialize embeddings
        this.userEmbedding = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05)
        );
        
        if (useDeepLearning) {
            // With Deep Learning: Use MLP for item tower with genre features
            this.itemEmbedding = null; // We'll use MLP instead
            this.setupMLPItemTower(numItems, embeddingDim);
        } else {
            // Without Deep Learning: Simple embedding lookup
            this.itemEmbedding = tf.variable(
                tf.randomNormal([numItems, embeddingDim], 0, 0.05)
            );
        }
        
        // Setup optimizer
        this.optimizer = tf.train.adam(0.001);
    }
    
    setupMLPItemTower(numItems, embeddingDim) {
        // MLP with one hidden layer for item tower
        const genreDim = 19; // Number of genre features
        
        // Hidden layer weights and biases
        this.mlpHiddenWeights = tf.variable(
            tf.randomNormal([genreDim, 64], 0, 0.05)
        );
        this.mlpHiddenBias = tf.variable(
            tf.zeros([64])
        );
        
        // Output layer weights and biases
        this.mlpOutputWeights = tf.variable(
            tf.randomNormal([64, embeddingDim], 0, 0.05)
        );
        this.mlpOutputBias = tf.variable(
            tf.zeros([embeddingDim])
        );
    }
    
    userForward(userIndices) {
        // User tower: simple embedding lookup
        return tf.gather(this.userEmbedding, userIndices);
    }
    
    itemForward(itemIndices, itemFeatures = null) {
        if (this.useDeepLearning && itemFeatures) {
            // Deep Learning: Use MLP with genre features
            return this.mlpItemForward(itemIndices, itemFeatures);
        } else {
            // Without Deep Learning: simple embedding lookup
            return tf.gather(this.itemEmbedding, itemIndices);
        }
    }
    
    mlpItemForward(itemIndices, itemFeatures) {
        return tf.tidy(() => {
            // Get genre features for the specified items
            const features = tf.tensor2d(
                itemIndices.arraySync().map(i => itemFeatures[i])
            );
            
            // MLP forward pass with one hidden layer
            const hidden = tf.relu(
                tf.add(tf.matMul(features, this.mlpHiddenWeights), this.mlpHiddenBias)
            );
            
            const output = tf.add(
                tf.matMul(hidden, this.mlpOutputWeights), 
                this.mlpOutputBias
            );
            
            return output;
        });
    }
    
    score(userEmbeddings, itemEmbeddings) {
        // Dot product scoring
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    computeLoss(userEmbeddings, positiveItemEmbeddings, negativeItemEmbeddings = null) {
        if (this.useDeepLearning) {
            // Use in-batch sampled softmax loss for deep learning approach
            return this.inBatchSoftmaxLoss(userEmbeddings, positiveItemEmbeddings);
        } else {
            // Use simple contrastive loss for non-deep learning approach
            return this.contrastiveLoss(userEmbeddings, positiveItemEmbeddings);
        }
    }
    
    inBatchSoftmaxLoss(userEmbeddings, positiveItemEmbeddings) {
        return tf.tidy(() => {
            // Compute logits: U @ I^T
            const logits = tf.matMul(userEmbeddings, positiveItemEmbeddings, false, true);
            
            // Labels are diagonal (each user's positive item)
            const batchSize = userEmbeddings.shape[0];
            const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
            
            // Softmax cross entropy loss
            const loss = tf.losses.softmaxCrossEntropy(labels, logits);
            return loss;
        });
    }
    
    contrastiveLoss(userEmbeddings, positiveItemEmbeddings) {
        return tf.tidy(() => {
            // Positive scores
            const positiveScores = this.score(userEmbeddings, positiveItemEmbeddings);
            
            // Sample negative items
            const negativeIndices = Array.from(
                {length: userEmbeddings.shape[0]}, 
                () => Math.floor(Math.random() * this.numItems)
            );
            
            let negativeItemEmbeddings;
            if (this.useDeepLearning) {
                // For MLP, we'd need to compute negative item embeddings
                // For simplicity, we'll use random embeddings in this case
                negativeItemEmbeddings = tf.randomNormal(
                    positiveItemEmbeddings.shape, 0, 0.05
                );
            } else {
                negativeItemEmbeddings = tf.gather(this.itemEmbedding, negativeIndices);
            }
            
            const negativeScores = this.score(userEmbeddings, negativeItemEmbeddings);
            
            // Margin-based contrastive loss
            const margin = 1.0;
            const loss = tf.relu(tf.sub(margin, tf.sub(positiveScores, negativeScores)));
            return tf.mean(loss);
        });
    }
    
    async trainEpoch(userIndices, itemIndices, batchSize = 512, itemFeatures = null) {
        const numBatches = Math.ceil(userIndices.length / batchSize);
        let totalLoss = 0;
        
        for (let i = 0; i < numBatches; i++) {
            const start = i * batchSize;
            const end = Math.min(start + batchSize, userIndices.length);
            
            const batchUserIndices = userIndices.slice(start, end);
            const batchItemIndices = itemIndices.slice(start, end);
            
            const loss = await this.trainBatch(batchUserIndices, batchItemIndices, itemFeatures);
            totalLoss += loss;
            
            // Clean up memory
            tf.engine().startScope();
            tf.engine().endScope();
        }
        
        return totalLoss / numBatches;
    }
    
    async trainBatch(userIndices, itemIndices, itemFeatures = null) {
        return tf.tidy(() => {
            const userIdxTensor = tf.tensor1d(userIndices, 'int32');
            const itemIdxTensor = tf.tensor1d(itemIndices, 'int32');
            
            // Get batch features for MLP if using deep learning
            let batchItemFeatures = null;
            if (this.useDeepLearning && itemFeatures) {
                batchItemFeatures = itemIndices.map(i => itemFeatures[i]);
            }
            
            const loss = this.optimizer.minimize(() => {
                const userEmbs = this.userForward(userIdxTensor);
                const itemEmbs = this.itemForward(itemIdxTensor, batchItemFeatures);
                
                return this.computeLoss(userEmbs, itemEmbs);
            }, true);
            
            return loss ? loss.dataSync()[0] : 0;
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward(tf.tensor1d([userIndex], 'int32'));
        });
    }
    
    getItemEmbeddings(itemIndices) {
        return tf.tidy(() => {
            const embeddings = this.itemForward(tf.tensor1d(itemIndices, 'int32'));
            return embeddings.arraySync();
        });
    }
    
    getScoresForAllItems(userEmbedding) {
        return tf.tidy(() => {
            if (this.useDeepLearning) {
                // For MLP approach, we need to compute all item embeddings
                // This is computationally expensive, so we do it in batches
                const batchSize = 100;
                const allScores = [];
                
                for (let i = 0; i < this.numItems; i += batchSize) {
                    const end = Math.min(i + batchSize, this.numItems);
                    const batchIndices = Array.from({length: end - i}, (_, j) => i + j);
                    
                    const itemEmbs = this.itemForward(tf.tensor1d(batchIndices, 'int32'));
                    const batchScores = this.score(
                        userEmbedding.tile([batchIndices.length, 1]), 
                        itemEmbs
                    );
                    
                    allScores.push(...batchScores.dataSync());
                }
                
                return allScores;
            } else {
                // Simple embedding lookup for non-DL approach
                const scores = tf.matMul(userEmbedding, this.itemEmbedding, false, true);
                return scores.dataSync();
            }
        });
    }
}
