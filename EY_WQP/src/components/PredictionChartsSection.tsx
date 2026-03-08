import { useMemo } from "react";
import { predictions, predictionStats } from "@/data/predictionData";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell,
} from "recharts";

// Build histogram bins from real prediction data
const buildHistogram = (values: number[], bins: number = 12) => {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / bins;
  const histogram = Array.from({ length: bins }, (_, i) => ({
    range: `${Math.round(min + i * binWidth)}`,
    count: 0,
    binStart: min + i * binWidth,
    binEnd: min + (i + 1) * binWidth,
  }));
  values.forEach((v) => {
    const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
    histogram[idx].count++;
  });
  return histogram;
};

const PredictionChartsSection = () => {
  const taHist = useMemo(() => buildHistogram(predictions.map((p) => p.alkalinity)), []);
  const ecHist = useMemo(() => buildHistogram(predictions.map((p) => p.conductance)), []);
  const drpHist = useMemo(() => buildHistogram(predictions.map((p) => p.phosphorus)), []);

  // Scatter: Alkalinity vs Conductance colored by DRP
  const scatterData = useMemo(
    () => predictions.map((p) => ({ alkalinity: p.alkalinity, conductance: p.conductance, drp: p.phosphorus })),
    []
  );

  return (
    <section id="predictions" className="py-24 bg-background relative">
      <div className="absolute inset-0 grid-dots opacity-30" />
      <div className="container mx-auto px-6 relative z-10">
        <div className="flex items-center gap-2 mb-6">
          <span className="text-xs font-mono text-primary/60">07</span>
          <div className="h-px flex-1 bg-border" />
          <span className="text-xs font-mono text-primary uppercase tracking-widest">Prediction Analysis</span>
          <div className="h-px flex-1 bg-border" />
        </div>

        <div className="text-center mb-12">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
            200 Test Predictions
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-sm">
            Distribution and relationships in the model's output on unseen test data.
          </p>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-3 mb-8">
          {[
            { label: "Total Alkalinity", mean: predictionStats.alkalinity.mean, min: predictionStats.alkalinity.min, max: predictionStats.alkalinity.max, unit: "mg/L" },
            { label: "Elec. Conductance", mean: predictionStats.conductance.mean, min: predictionStats.conductance.min, max: predictionStats.conductance.max, unit: "mS/m" },
            { label: "Dissolved Reactive P", mean: predictionStats.phosphorus.mean, min: predictionStats.phosphorus.min, max: predictionStats.phosphorus.max, unit: "µg/L" },
          ].map((s) => (
            <div key={s.label} className="rounded-lg border border-border bg-card p-4">
              <h4 className="font-semibold text-foreground text-xs mb-2">{s.label}</h4>
              <div className="text-lg font-bold font-mono text-primary">{s.mean} <span className="text-xs text-muted-foreground font-normal">{s.unit}</span></div>
              <div className="text-xs text-muted-foreground font-mono mt-1">
                range: {s.min} — {s.max}
              </div>
            </div>
          ))}
        </div>

        {/* Distribution histograms */}
        <div className="grid md:grid-cols-3 gap-4 mb-8">
          {[
            { title: "Total Alkalinity Distribution", data: taHist, color: "hsl(168, 80%, 46%)" },
            { title: "Elec. Conductance Distribution", data: ecHist, color: "hsl(200, 80%, 55%)" },
            { title: "DRP Distribution", data: drpHist, color: "hsl(142, 70%, 45%)" },
          ].map((chart) => (
            <Card key={chart.title} className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="text-xs font-mono text-foreground">{chart.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chart.data} margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="range" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} angle={-45} textAnchor="end" height={40} />
                    <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
                    <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "0.5rem", color: "hsl(var(--foreground))" }} />
                    <Bar dataKey="count" fill={chart.color} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Scatter: TA vs EC */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-mono text-foreground">alkalinity_vs_conductance(color=drp)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="alkalinity"
                  name="Total Alkalinity"
                  type="number"
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  label={{ value: "Total Alkalinity (mg/L)", position: "bottom", fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                />
                <YAxis
                  dataKey="conductance"
                  name="Conductance"
                  type="number"
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  label={{ value: "Conductance (mS/m)", angle: -90, position: "insideLeft", fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "0.5rem", color: "hsl(var(--foreground))" }}
                  formatter={(value: number, name: string) => [value.toFixed(1), name]}
                />
                <Scatter data={scatterData} fillOpacity={0.8}>
                  {scatterData.map((entry, i) => {
                    // Color by DRP: low=teal, high=orange
                    const ratio = (entry.drp - predictionStats.phosphorus.min) / (predictionStats.phosphorus.max - predictionStats.phosphorus.min);
                    const hue = 168 - ratio * 140; // teal(168) to orange(28)
                    return <Cell key={i} fill={`hsl(${hue}, 75%, 50%)`} />;
                  })}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            <p className="text-xs text-muted-foreground text-center mt-2 font-mono">
              Color: teal = low DRP → orange = high DRP
            </p>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default PredictionChartsSection;
