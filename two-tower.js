// two-tower.js
class TwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, useDeepLearning = false, numGenres = 0) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.useDeepLearning = useDeepLearning;
        this.numGenres = numGenres;
        
        // Initialize embeddings with proper Xavier/Glorot initialization
        const userInit = tf.initializers.glorotNormal();
        const itemInit = tf.initializers.glorotNormal();
        
        this.userEmbedding = tf.variable(
            userInit.apply([numUsers, embeddingDim]), 
            true, 
            'userEmbedding'
        );
        
        this.itemEmbedding = tf.variable(
            itemInit.apply([numItems, embeddingDim]), 
            true, 
            'itemEmbedding'
        );
        
        // MLP layers for deep learning version
        if (useDeepLearning && numGenres > 0) {
            // User tower MLP
            this.userHidden = tf.layers.dense({
                units: 64,
                activation: 'relu',
                kernelInitializer: 'glorotNormal'
            });
            
            this.userOutput = tf.layers.dense({
                units: embeddingDim,
                activation: 'linear',
                kernelInitializer: 'glorotNormal'
            });
            
            // Item tower MLP with genre features
            this.itemHidden = tf.layers.dense({
                units: 64,
                activation: 'relu',
                kernelInitializer: 'glorotNormal'
            });
            
            this.itemOutput = tf.layers.dense({
                units: embeddingDim,
                activation: 'linear',
                kernelInitializer: 'glorotNormal'
            });
            
            // Genre embedding (if we want to treat genres as embeddings)
            this.genreEmbedding = tf.variable(
                tf.randomNormal([numGenres, 8], 0, 0.05),
                true,
                'genreEmbedding'
            );
        }
        
        this.optimizer = tf.train.adam(0.001);
    }

    /**
     * User tower: transforms user ID to embedding
     * For basic model: simple embedding lookup
     * For DL model: MLP transformation of embedding
     */
    userForward(userIndices) {
        const userEmbs = tf.gather(this.userEmbedding, userIndices);
        
        if (this.useDeepLearning) {
            // MLP transformation for user tower
            let hidden = this.userHidden.apply(userEmbs);
            return this.userOutput.apply(hidden);
        }
        
        return userEmbs;
    }

    /**
     * Item tower: transforms item ID to embedding
     * For basic model: simple embedding lookup  
     * For DL model: MLP transformation combining item embedding and genre features
     */
    itemForward(itemIndices, itemsMap = null) {
        const itemEmbs = tf.gather(this.itemEmbedding, itemIndices);
        
        if (this.useDeepLearning && itemsMap && this.numGenres > 0) {
            // Get genre features for items
            const genreFeatures = this.getGenreFeatures(itemIndices, itemsMap);
            
            // Combine item embedding with genre features
            const combined = tf.concat([itemEmbs, genreFeatures], -1);
            
            // MLP transformation for item tower
            let hidden = this.itemHidden.apply(combined);
            return this.itemOutput.apply(hidden);
        }
        
        return itemEmbs;
    }

    /**
     * Extract genre features for items
     * Converts genre flags to dense genre embeddings
     */
    getGenreFeatures(itemIndices, itemsMap) {
        const batchSize = itemIndices.shape[0];
        const genreEmbeddings = [];
        
        // Process each item in batch
        for (let i = 0; i < batchSize; i++) {
            const itemIdx = itemIndices.dataSync()[i];
            const itemId = Array.from(itemsMap.keys())[itemIdx];
            const item = itemsMap.get(itemId);
            
            if (item && item.genreFlags) {
                // Average genre embeddings for this item's genres
                const genreIndices = item.genreFlags
                    .map((flag, idx) => flag === 1 ? idx : -1)
                    .filter(idx => idx !== -1);
                
                if (genreIndices.length > 0) {
                    const genreEmbs = tf.gather(this.genreEmbedding, genreIndices);
                    const avgGenreEmb = tf.mean(genreEmbs, 0);
                    genreEmbeddings.push(avgGenreEmb);
                } else {
                    // No genres - use zeros
                    genreEmbeddings.push(tf.zeros([8]));
                }
            } else {
                // No genre info - use zeros
                genreEmbeddings.push(tf.zeros([8]));
            }
        }
        
        return tf.stack(genreEmbeddings);
    }

    /**
     * Score function: dot product between user and item embeddings
     * Measures compatibility between user preferences and item characteristics
     */
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }

    /**
     * In-batch sampled softmax loss
     * For each user-positive item pair, treat all other items in batch as negatives
     * This is efficient and works well for retrieval tasks
     */
    computeLoss(userEmbeddings, itemEmbeddings) {
        // Compute scores matrix: batch_size x batch_size
        const scores = tf.matMul(userEmbeddings, itemEmbeddings, false, true);
        
        // Labels are diagonal (each user matches with their positive item)
        const batchSize = userEmbeddings.shape[0];
        const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
        
        // Softmax cross entropy loss
        const loss = tf.losses.softmaxCrossEntropy(labels, scores);
        
        return loss;
    }

    /**
     * Training step for one batch
     * Uses gradient tape to track operations for automatic differentiation
     */
    async trainStep(userIndices, positiveItemIndices, itemsMap) {
        return tf.tidy(() => {
            const userEmbeddings = this.userForward(userIndices);
            const itemEmbeddings = this.itemForward(positiveItemIndices, itemsMap);
            
            const loss = this.computeLoss(userEmbeddings, itemEmbeddings);
            return loss;
        });
    }

    /**
     * Train for one epoch using in-batch negatives
     * Processes data in batches for memory efficiency
     */
    async trainEpoch(interactions, userMap, itemMap, batchSize, itemsMap) {
        // Shuffle interactions
        const shuffled = [...interactions].sort(() => Math.random() - 0.5);
        let totalLoss = 0;
        let batchCount = 0;
        
        for (let i = 0; i < shuffled.length; i += batchSize) {
            const batch = shuffled.slice(i, i + batchSize);
            
            // Convert to indices
            const userIndices = tf.tensor1d(
                batch.map(interaction => userMap.get(interaction.userId)),
                'int32'
            );
            const itemIndices = tf.tensor1d(
                batch.map(interaction => itemMap.get(interaction.itemId)),
                'int32'
            );
            
            const loss = this.trainStep(userIndices, itemIndices, itemsMap);
            const gradients = tf.grad(loss => loss)({
                userEmbedding: this.userEmbedding,
                itemEmbedding: this.itemEmbedding
            });
            
            this.optimizer.applyGradients([
                { tensor: gradients.userEmbedding, variable: this.userEmbedding },
                { tensor: gradients.itemEmbedding, variable: this.itemEmbedding }
            ]);
            
            totalLoss += loss.dataSync()[0];
            batchCount++;
            
            // Cleanup
            tf.dispose([userIndices, itemIndices, loss, gradients]);
            
            // Prevent memory buildup
            if (batchCount % 10 === 0) {
                await tf.nextFrame();
            }
        }
        
        return totalLoss / batchCount;
    }

    /**
     * Get user embedding for inference
     */
    async getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            const userIdxTensor = tf.tensor1d([userIndex], 'int32');
            return this.userForward(userIdxTensor);
        });
    }

    /**
     * Get all item embeddings for inference
     */
    async getItemEmbeddings(itemIndices) {
        return tf.tidy(() => {
            const itemIdxTensor = tf.tensor1d(itemIndices, 'int32');
            return this.itemForward(itemIdxTensor);
        });
    }

    /**
     * Compute scores for a user against all items
     * Uses batching to avoid memory issues with large item sets
     */
    async getScoresForAllItems(userEmbedding, allItemIndices, batchSize = 1000) {
        const scores = [];
        
        for (let i = 0; i < allItemIndices.length; i += batchSize) {
            const batchIndices = allItemIndices.slice(i, i + batchSize);
            const batchTensor = tf.tensor1d(batchIndices, 'int32');
            
            const batchEmbeddings = this.itemForward(batchTensor);
            const batchScores = this.score(
                userEmbedding.tile([batchIndices.length, 1]),
                batchEmbeddings
            );
            
            scores.push(...await batchScores.data());
            
            tf.dispose([batchTensor, batchEmbeddings, batchScores]);
        }
        
        return scores;
    }
}
