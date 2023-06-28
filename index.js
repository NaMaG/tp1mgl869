/*const LogisticRegression = require('logistic-regression');

const regression = new LogisticRegression();

*/

// Importer la bibliothèque TensorFlow.js
const tf = require('@tensorflow/tfjs');

// Définir les données d'entraînement
const xTrain = tf.tensor2d([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]);
const yTrain = tf.tensor2d([[0], [0], [0], [1], [1]]);

// Définir le modèle de régression logistique
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' }));

// Compiler le modèle
model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam' });

// Entraîner le modèle
model.fit(xTrain, yTrain, { epochs: 100 })
  .then(() => {
    console.log('Entraînement terminé.');

    // Définir les données de test
    const xTest = tf.tensor2d([[6, 6], [7, 7], [8, 8]]);
    
    // Prédire les résultats
    const predictions = model.predict(xTest);
    predictions.print();
  })
  .catch((error) => {
    console.error('Une erreur s\'est produite lors de l\'entraînement du modèle :', error);
  });
