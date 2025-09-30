import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell } from "recharts";

export default function RequestsChart({ data }) {
  let apiError = data.filter(m => m.status === "API Error").length;
  let internalError = data.filter(m => m.status === "Internal Server Error").length;
  let good = data.length - internalError - apiError;

  const chartData = [
    { name: "Good Requests", value: good, color: "#82ca9d" },
    { name: "Internal Server Errors", value: internalError, color: "#FFA500"},
    { name: "API Errors", value: apiError, color: "#ff6b6b" }
  ];

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const color = payload[0].payload.color;
      return (
        <div
          style={{
            backgroundColor: "#fff",
            border: "1px solid #ccc",
            padding: "8px",
            borderRadius: "4px",
            color: color // Tooltip text color matches bar color
          }}
        >
          <p>{label}</p>
          <p>{`Requests: ${payload[0].value}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white p-4 rounded-2xl shadow">
      <h2 className="text-lg font-semibold mb-2">Total Requests</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 20, right: 20, bottom: 20, left: 15 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" allowDecimals={false} />
          <YAxis type="category" dataKey="name" />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="value" name="Requests">
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
