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
                
                if (genreInd
