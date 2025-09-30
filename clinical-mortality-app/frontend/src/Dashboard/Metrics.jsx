import { useEffect, useState } from "react";
import axios from "axios";
import LatencyChart from "./components/LatencyCharts";
import RequestsChart from "./components/RequestsChart";
import LatestRequestCard from "./components/LatestRequestCard";

function Metrics() {
  const [metrics, setMetrics] = useState([]);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await axios.get("http://localhost:8000/metrics");
        setMetrics(res.data.reverse()); // oldest first
      } catch (err) {
        console.error("Failed to fetch metrics", err);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000); // refresh every 5s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-2xl font-bold text-white mb-6">API Metrics Dashboard</h2>
        <div className="space-y-6">
          <LatencyChart data={metrics} />
          <RequestsChart data={metrics} />
          <LatestRequestCard data={metrics} />
        </div>
      </div>
    </div>
  );
}

export default Metrics
