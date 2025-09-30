export default function LatestRequestCard({ data }) {
  if (!data.length) return null;

  const latest = data[data.length - 1];
  const isOk = latest.status === "success" || latest.status === "200";

  return (

        <div className="bg-white p-4 rounded-2xl shadow">
          <h2 className="text-lg font-semibold mb-2">Latest Request</h2>
          <p>
            <strong>Status:</strong>{" "}
            <span className={isOk ? "text-green-600" : "text-red-600"}>
              {latest.status}
            </span>
          </p>
          <p>
            <strong>Latency:</strong> {latest.latency.toFixed(2)} ms
          </p>
          <p>
            <strong>Time:</strong> {new Date(latest.timestamp).toLocaleString()}
          </p>
        </div>

  );
}
