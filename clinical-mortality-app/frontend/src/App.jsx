import { useState } from 'react'
import axios from 'axios'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Metrics from './Dashboard/Metrics'

function Prediction() {
  const [formData, setFormData] = useState({
    age: '',
    sex: 'Male',
    bmi: '',
    systolic_bp: '',
    diastolic_bp: '',
    glucose: '',
    cholesterol: '',
    creatinine: '',
    diabetes: 0,
    hypertension: 0,
    diagnosis: 'Pneumonia',
    readmission_30d: 0
  })

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const diagnosisList = [
    'Pneumonia',
    'Heart Failure',
    'Myocardial Infarction',
    'Stroke',
    'Sepsis',
    'COPD',
    'Diabetes Complications',
    'Renal Failure'
  ]

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      // Convertir les valeurs en nombres
      const payload = {
        ...formData,
        age: parseInt(formData.age),
        bmi: parseFloat(formData.bmi),
        systolic_bp: parseInt(formData.systolic_bp),
        diastolic_bp: parseInt(formData.diastolic_bp),
        glucose: parseFloat(formData.glucose),
        cholesterol: parseFloat(formData.cholesterol),
        creatinine: parseFloat(formData.creatinine)
      }
      console.log('Submitting payload:', payload)
      const response = await axios.post('http://localhost:8000/predict', payload)
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Une erreur est survenue lors de la prédiction')
    } finally {
      setLoading(false)
    }
  }

  // Calculer le pourcentage de risque
  const riskPercentage = prediction ? Math.round(prediction.result.probas["1"] * 100) : 0
  const isHighRisk = riskPercentage >= 50
  const predictionResult = prediction?.result.prediction

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-4">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
            </svg>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">
            Prédiction de Mortalité Clinique
          </h1>
          <p className="text-gray-300 text-lg">
            Évaluez le risque de mortalité basé sur les données cliniques du patient
          </p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 shadow-2xl overflow-hidden">
          <div className="p-8">
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Section Informations de base */}
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <div className="w-2 h-6 bg-blue-500 rounded-full mr-3"></div>
                  Informations de base
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-200">
                      Âge
                    </label>
                    <input
                      type="number"
                      name="age"
                      value={formData.age}
                      onChange={handleChange}
                      required
                      min="0"
                      max="120"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                      placeholder="Ex: 45"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-200">
                      Sexe
                    </label>
                    <select
                      name="sex"
                      value={formData.sex}
                      onChange={handleChange}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    >
                      <option value="Male" className="text-gray-900">Homme</option>
                      <option value="Female" className="text-gray-900">Femme</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-200">
                      IMC (BMI)
                    </label>
                    <input
                      type="number"
                      name="bmi"
                      value={formData.bmi}
                      onChange={handleChange}
                      required
                      step="0.1"
                      min="10"
                      max="60"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                      placeholder="Ex: 24.5"
                    />
                  </div>
                </div>
              </div>

              {/* Section Tension artérielle */}
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <div className="w-2 h-6 bg-purple-500 rounded-full mr-3"></div>
                  Tension artérielle
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-200">
                      Tension systolique
                    </label>
                    <input
                      type="number"
                      name="systolic_bp"
                      value={formData.systolic_bp}
                      onChange={handleChange}
                      required
                      min="60"
                      max="250"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                      placeholder="Ex: 120"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-200">
                      Tension diastolique
                    </label>
                    <input
                      type="number"
                      name="diastolic_bp"
                      value={formData.diastolic_bp}
                      onChange={handleChange}
                      required
                      min="40"
                      max="150"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                      placeholder="Ex: 80"
                    />
                  </div>
                </div>
              </div>

              {/* Section Analyses biologiques */}
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <div className="w-2 h-6 bg-green-500 rounded-full mr-3"></div>
                  Analyses biologiques
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-200">
                      Glucose (mg/dL)
                    </label>
                    <input
                      type="number"
                      name="glucose"
                      value={formData.glucose}
                      onChange={handleChange}
                      required
                      step="0.1"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                      placeholder="Ex: 95.0"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-200">
                      Cholestérol (mg/dL)
                    </label>
                    <input
                      type="number"
                      name="cholesterol"
                      value={formData.cholesterol}
                      onChange={handleChange}
                      required
                      step="0.1"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                      placeholder="Ex: 185.0"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-200">
                      Créatinine (mg/dL)
                    </label>
                    <input
                      type="number"
                      name="creatinine"
                      value={formData.creatinine}
                      onChange={handleChange}
                      required
                      step="0.01"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                      placeholder="Ex: 0.85"
                    />
                  </div>
                </div>
              </div>

              {/* Section Diagnostic et antécédents */}
              <div className="space-y-6">
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold text-white flex items-center">
                    <div className="w-2 h-6 bg-orange-500 rounded-full mr-3"></div>
                    Diagnostic principal
                  </h2>
                  <select
                    name="diagnosis"
                    value={formData.diagnosis}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  >
                    {diagnosisList.map(d => (
                      <option key={d} value={d} className="text-gray-900">{d}</option>
                    ))}
                  </select>
                </div>

                <div className="space-y-4">
                  <h2 className="text-xl font-semibold text-white flex items-center">
                    <div className="w-2 h-6 bg-red-500 rounded-full mr-3"></div>
                    Antécédents médicaux
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <label className="flex items-center space-x-3 p-4 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 transition-colors duration-200 cursor-pointer">
                      <input
                        type="checkbox"
                        name="diabetes"
                        checked={formData.diabetes === 1}
                        onChange={handleChange}
                        className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500 bg-white/10 border-white/20"
                      />
                      <span className="text-white font-medium">Diabète</span>
                    </label>

                    <label className="flex items-center space-x-3 p-4 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 transition-colors duration-200 cursor-pointer">
                      <input
                        type="checkbox"
                        name="hypertension"
                        checked={formData.hypertension === 1}
                        onChange={handleChange}
                        className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500 bg-white/10 border-white/20"
                      />
                      <span className="text-white font-medium">Hypertension</span>
                    </label>

                    <label className="flex items-center space-x-3 p-4 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 transition-colors duration-200 cursor-pointer">
                      <input
                        type="checkbox"
                        name="readmission_30d"
                        checked={formData.readmission_30d === 1}
                        onChange={handleChange}
                        className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500 bg-white/10 border-white/20"
                      />
                      <span className="text-white font-medium">Réadmission 30j</span>
                    </label>
                  </div>
                </div>
              </div>

              {/* Bouton de soumission */}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-6 rounded-xl font-bold text-lg hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-4 focus:ring-blue-500/50 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.02] disabled:hover:scale-100 shadow-lg"
              >
                {loading ? (
                  <div className="flex items-center justify-center">
                    <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin mr-3"></div>
                    Calcul en cours...
                  </div>
                ) : (
                  'Obtenir la prédiction de risque'
                )}
              </button>
            </form>

            {/* Affichage des erreurs */}
            {error && (
              <div className="mt-8 p-6 bg-red-500/10 border border-red-500/20 rounded-2xl backdrop-blur-sm">
                <div className="flex items-center space-x-3 text-red-300 mb-2">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="font-semibold text-lg">Erreur</p>
                </div>
                <p className="text-red-200">{error}</p>
              </div>
            )}

            {/* Affichage de la prédiction */}
            {prediction && (
              <div className="mt-8 p-8 bg-gradient-to-r from-slate-800 to-purple-900/50 rounded-2xl border border-white/10 backdrop-blur-sm">
                <h2 className="text-2xl font-bold text-white mb-6 text-center">
                  Résultat de l'analyse
                </h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                  {/* Cercle de progression */}
                  <div className="flex justify-center">
                    <div className="relative w-64 h-64">
                      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                        {/* Cercle de fond */}
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="none"
                          className="text-white/10"
                        />
                        {/* Cercle de progression */}
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="none"
                          strokeLinecap="round"
                          strokeDasharray="251.2"
                          strokeDashoffset={251.2 - (251.2 * riskPercentage) / 100}
                          className={`transition-all duration-1000 ease-out ${
                            isHighRisk ? 'text-red-500' : 'text-green-500'
                          }`}
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className={`text-4xl font-bold ${
                          isHighRisk ? 'text-red-400' : 'text-green-400'
                        }`}>
                          {riskPercentage}%
                        </span>
                        <span className="text-white/60 text-sm mt-2">Risque de mortalité</span>
                      </div>
                    </div>
                  </div>

                  {/* Détails de la prédiction */}
                  <div className="space-y-6">
                    <div className="bg-white/5 rounded-xl p-6 border border-white/10">
                      <h3 className="text-lg font-semibold text-white mb-4">Résultat de la prédiction</h3>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-white/70">Prédiction :</span>
                          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                            predictionResult === "0" 
                              ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                              : 'bg-red-500/20 text-red-300 border border-red-500/30'
                          }`}>
                            {predictionResult === "0" ? 'Faible risque' : 'Risque élevé'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-white/70">Probabilité risque :</span>
                          <span className="text-white font-semibold">
                            {(prediction.result.probas["1"] * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-white/70">Probabilité sécurité :</span>
                          <span className="text-white font-semibold">
                            {(prediction.result.probas["0"] * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-white/5 rounded-xl p-6 border border-white/10">
                      <h3 className="text-lg font-semibold text-white mb-3">Recommandation</h3>
                      <p className="text-white/80 text-sm leading-relaxed">
                        {isHighRisk 
                          ? 'Surveillance médicale rapprochée recommandée. Considérer une intervention préventive.'
                          : 'Surveillance standard recommandée. Maintenir le suivi médical régulier.'
                        }
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-white/40 text-sm">
            Outil d'aide à la décision clinique - Utilisation médicale 
          </p>
          <p className="text-white/40 text-sm">
            Les résultats fournis sont issue d'un modèle de machine learning et ne remplacent pas l'avis d'un professionnel de santé.
          </p>
          <p className="text-white/40 text-sm">
            © 2025 - Tous droits réservés - Groupe MLOps EPITA 
          </p>
        </div>
      </div>
    </div>
  )
}

function App() {
  return (
    <Router>
      <nav className="p-4 flex gap-4 bg-slate-900 text-white">
        <Link to="/">Prédiction</Link>
        <Link to="/metrics">Métriques</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Prediction />} />
        <Route path="/metrics" element={<Metrics />} />
      </Routes>
    </Router>
  )
}

export default App