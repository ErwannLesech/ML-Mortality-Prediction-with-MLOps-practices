export default function LatestRequestCard({ data }) {
  if (!data.length) return null;

  const latest = data[data.length - 1];
  const isOk = latest.status === "success" || latest.status === "200";

  return (
    <div
      className={`p-4 rounded-2xl shadow ${
        isOk ? "bg-green-500" : "bg-red-500"
      } text-white`}
    >
      <h2 className="text-lg font-semibold mb-2">Latest Request</h2>
      <p>
        <strong>Status:</strong>{" "}
        <span className="font-bold">
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
