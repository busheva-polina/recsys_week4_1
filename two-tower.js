// two-tower.js
class TwoTowerModel {
    constructor(config) {
        this.config = {
            numUsers: 943,
            numItems: 1682,
            embeddingDim: 32,
            learningRate: 0.001,
            epochs: 50,
            batchSize: 1024,
            useMLP: true,
            mlpHiddenUnits: 64,
            ...config
        };
        
        this.userEmbedding = null;
        this.itemEmbedding = null;
        this.mlpLayers = null;
        this.optimizer = null;
        this.isInitialized = false;
        
        this.initializeModel();
    }

    initializeModel() {
        // Two-tower model: user tower (embedding) and item tower (MLP with genre features)
        const { numUsers, numItems, embeddingDim, useMLP, mlpHiddenUnits } = this.config;
        
        // User tower: simple embedding lookup
        this.userEmbedding = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05)
        );
        
        // Item tower: either simple embedding or MLP with genre features
        if (useMLP) {
            // MLP with one hidden layer for item tower
            this.itemEmbedding = tf.variable(
                tf.randomNormal([numItems, embeddingDim], 0, 0.05)
            );
            
            // MLP layers for processing genre features
            this.mlpLayers = {
                hidden: tf.layers.dense({
                    units: mlpHiddenUnits,
                    activation: 'relu',
                    inputShape: [19] // 19 genre features
                }),
                output: tf.layers.dense({
                    units: embeddingDim,
                    activation: 'linear'
                })
            };
        } else {
            // Simple item embedding (non-MLP baseline)
            this.itemEmbedding = tf.variable(
                tf.randomNormal([numItems, embeddingDim], 0, 0.05)
            );
        }
        
        this.optimizer = tf.train.adam(this.config.learningRate);
        this.isInitialized = true;
    }

    userForward(userIndices) {
        // User tower: embedding lookup
        return tf.gather(this.userEmbedding, userIndices);
    }

    async itemForward(itemIndices, genreFeatures = null) {
        // Item tower: either embedding lookup or MLP with genre features
        if (this.config.useMLP && genreFeatures) {
            // Use MLP to process genre features
            const genreTensor = tf.tensor2d(genreFeatures);
            const hidden = this.mlpLayers.hidden.apply(genreTensor);
            const mlpOutput = this.mlpLayers.output.apply(hidden);
            
            // Combine MLP output with item embedding
            const itemEmb = tf.gather(this.itemEmbedding, itemIndices);
            return tf.add(mlpOutput, itemEmb); // Residual connection
            
        } else {
            // Simple embedding lookup
            return tf.gather(this.itemEmbedding, itemIndices);
        }
    }

    score(userEmbeddings, itemEmbeddings) {
        // Dot product scoring: u · i
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }

    computeLoss(userEmbeddings, positiveItemEmbeddings, negativeItemEmbeddings = null) {
        // In-batch sampled softmax loss
        const batchSize = userEmbeddings.shape[0];
        const embeddingDim = userEmbeddings.shape[1];
        
        // Compute all pairwise scores: U @ I^T
        const scores = tf.matMul(userEmbeddings, positiveItemEmbeddings, false, true);
        
        // Labels are diagonal (each user's positive item)
        const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
        
        // Softmax cross entropy loss
        const loss = tf.losses.softmaxCrossEntropy(labels, scores);
        
        return loss;
    }

    async train(interactions, userIndexMap, itemIndexMap, progressCallback = null) {
        if (!this.isInitialized) {
            throw new Error('Model not initialized');
        }

        const { epochs, batchSize } = this.config;
        const lossHistory = [];
        
        // Convert interactions to indexed format
        const indexedInteractions = interactions.map(interaction => ({
            userIndex: userIndexMap.get(interaction.userId),
            itemIndex: itemIndexMap.get(interaction.itemId),
            rating: interaction.rating
        })).filter(interaction => 
            interaction.userIndex !== undefined && 
            interaction.itemIndex !== undefined
        );

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let batchCount = 0;
            
            // Shuffle interactions
            const shuffled = indexedInteractions.sort(() => Math.random() - 0.5);
            
            // Process in batches
            for (let i = 0; i < shuffled.length; i += batchSize) {
                const batch = shuffled.slice(i, i + batchSize);
                if (batch.length === 0) continue;
                
                const loss = await this.trainBatch(batch);
                epochLoss += loss;
                batchCount++;
                
                // Clean up memory
                tf.engine().startScope();
                tf.engine().endScope();
            }
            
            const avgLoss = epochLoss / batchCount;
            lossHistory.push(avgLoss);
            
            if (progressCallback) {
                await progressCallback(epoch, avgLoss);
            }
            
            // Force garbage collection in browsers that support it
            if (tf.memory().numTensors > 1000) {
                console.log('Cleaning up memory:', tf.memory().numTensors, 'tensors');
                tf.engine().startScope();
                tf.engine().endScope();
            }
        }
        
        return lossHistory;
    }

    async trainBatch(batch) {
        return tf.tidy(() => {
            const userIndices = batch.map(interaction => interaction.userIndex);
            const itemIndices = batch.map(interaction => interaction.itemIndex);
            
            // Get genre features for items if using MLP
            let genreFeatures = null;
            if (this.config.useMLP) {
                genreFeatures = batch.map(interaction => {
                    // This would need access to items data - in practice you'd pass this in
                    // For now, using random features as placeholder
                    return new Array(19).fill(0).map(() => Math.random());
                });
            }
            
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            
            const loss = this.optimizer.minimize(() => {
                const userEmb = this.userForward(userTensor);
                const itemEmb = this.itemForward(itemTensor, genreFeatures);
                
                return this.computeLoss(userEmb, itemEmb);
            }, true);
            
            return loss ? loss.dataSync()[0] : 0;
        });
    }

    async getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            const userTensor = tf.tensor1d([userIndex], 'int32');
            return this.userForward(userTensor);
        });
    }

    async getItemEmbeddings(itemIndices) {
        return tf.tidy(() => {
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            return this.itemForward(itemTensor);
        });
    }

    async getRecommendations(userIndex, ratedItems, itemIndexMap, reverseItemIndex, items, topK = 10) {
        return tf.tidy(async () => {
            // Get user embedding
            const userEmb = await this.getUserEmbedding(userIndex);
            
            // Get all item embeddings (in batches for memory efficiency)
            const allItemIndices = Array.from(itemIndexMap.values());
            const batchSize = 512;
            const allScores = [];
            
            for (let i = 0; i < allItemIndices.length; i += batchSize) {
                const batchIndices = allItemIndices.slice(i, i + batchSize);
                const itemTensor = tf.tensor1d(batchIndices, 'int32');
                const itemEmbs = await this.itemForward(itemTensor);
                
                // Compute scores for this batch
                const userEmbsExpanded = userEmb.expandDims(1).tile([1, batchIndices.length, 1]);
                const itemEmbsExpanded = itemEmbs.expandDims(0);
                const batchScores = this.score(userEmbsExpanded, itemEmbsExpanded);
                
                const batchScoresArray = await batchScores.data();
                allScores.push(...batchScoresArray);
                
                // Clean up
                itemTensor.dispose();
                itemEmbs.dispose();
                userEmbsExpanded.dispose();
                itemEmbsExpanded.dispose();
                batchScores.dispose();
            }
            
            userEmb.dispose();
            
            // Combine scores with item information
            const scoredItems = allItemIndices.map((itemIndex, idx) => ({
                itemId: reverseItemIndex.get(itemIndex),
                score: allScores[idx],
                title: items.get(reverseItemIndex.get(itemIndex))?.title || 'Unknown'
            }));
            
            // Filter out rated items and get top K
            const unratedItems = scoredItems.filter(item => 
                !ratedItems.has(item.itemId)
            );
            
            return unratedItems
                .sort((a, b) => b.score - a.score)
                .slice(0, topK);
        });
    }

    dispose() {
        if (this.userEmbedding) this.userEmbedding.dispose();
        if (this.itemEmbedding) this.itemEmbedding.dispose();
        if (this.optimizer) this.optimizer.dispose();
    }
}
