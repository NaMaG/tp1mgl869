const LogisticRegression = require('ml-logistic-regression');
const csv = require('csvtojson');

// Charger les données à partir d'un fichier CSV
const loadData = async () => {
  const data = await csv().fromFile('donnees.csv');
  const features = data.map((row) => [Number(row.feature1), Number(row.feature2)]);
  const labels = data.map((row) => Number(row.label));
  return { features, labels };
};

// Entraîner le modèle de régression logistique
const trainModel = async () => {
  const { features, labels } = await loadData();
  const logisticRegression = new LogisticRegression({ numSteps: 10000, learningRate: 5e-3 });
  logisticRegression.train(features, labels);
  return logisticRegression;
};

// Prédire avec le modèle entraîné
const predict = (model, data) => {
  const predictions = model.predict(data);
  console.log('Prédictions :', predictions);
};

// Exécuter le programme
const run = async () => {
  const model = await trainModel();
  const newData = [[2, 3], [4, 5]]; // Données de test pour prédire
  predict(model, newData);
};

run();
