import { useState } from "react";
import { featureImportanceByTarget } from "@/data/modelData";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

const tabs = [
  { key: "Total Alkalinity", label: "Total Alkalinity", short: "TA", model: "XGBoost", r2: "0.85" },
  { key: "Electrical Conductance", label: "Electrical Conductance", short: "EC", model: "XGBoost", r2: "0.87" },
  { key: "Dissolved Reactive Phosphorus", label: "Dissolved Reactive Phosphorus", short: "DRP", model: "ExtraTrees", r2: "0.72" },
];

const barColors = [
  "hsl(168, 80%, 46%)", "hsl(168, 70%, 40%)", "hsl(168, 60%, 35%)",
  "hsl(200, 80%, 55%)", "hsl(200, 70%, 48%)", "hsl(200, 60%, 42%)",
  "hsl(168, 50%, 38%)", "hsl(200, 50%, 45%)", "hsl(168, 40%, 35%)", "hsl(200, 40%, 40%)",
];

const FeatureImportanceSection = () => {
  const [activeTab, setActiveTab] = useState(tabs[0].key);
  const activeTabData = tabs.find((t) => t.key === activeTab)!;
  const sorted = [...featureImportanceByTarget[activeTab]].sort((a, b) => b.importance - a.importance);

  return (
    <section id="features" className="py-24 bg-card/50 relative">
      <div className="absolute inset-0 grid-dots opacity-30" />
      <div className="container mx-auto px-6 relative z-10">
        <div className="flex items-center gap-2 mb-6">
          <span className="text-xs font-mono text-primary/60">06</span>
          <div className="h-px flex-1 bg-border" />
          <span className="text-xs font-mono text-primary uppercase tracking-widest">Feature Importance</span>
          <div className="h-px flex-1 bg-border" />
        </div>

        <div className="text-center mb-12">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
            What Drives Predictions?
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-sm">
            Geographic location and satellite spectral data dominate across all targets.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex justify-center gap-1 mb-6 flex-wrap">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-3 py-1.5 rounded text-xs font-mono transition-all ${
                activeTab === tab.key
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground border border-border"
              }`}
            >
              <span className="hidden md:inline">{tab.label}</span>
              <span className="md:hidden">{tab.short}</span>
            </button>
          ))}
        </div>

        <div className="text-center mb-8">
          <span className="text-xs text-muted-foreground font-mono">
            model=<span className="text-foreground">{activeTabData.model}</span> | 
            cv_r2=<span className="text-primary">{activeTabData.r2}</span>
          </span>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          <Card className="bg-card border-border">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-mono text-foreground">feature_importance.plot(top=10)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={sorted} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 100 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} domain={[0, 0.35]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                  <YAxis dataKey="feature" type="category" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} width={110} />
                  <Tooltip
                    formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                    contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "0.5rem", fontSize: "12px", color: "hsl(var(--foreground))" }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {sorted.map((_, i) => (
                      <Cell key={i} fill={barColors[i % barColors.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="space-y-2">
            {sorted.map((item, i) => (
              <div key={item.feature} className="flex items-start gap-3 p-3 rounded-lg border border-border bg-card hover:border-primary/20 transition-all">
                <div
                  className="flex-shrink-0 w-10 h-10 rounded-md flex items-center justify-center font-bold text-xs font-mono"
                  style={{ backgroundColor: barColors[i % barColors.length], color: "hsl(var(--primary-foreground))" }}
                >
                  {(item.importance * 100).toFixed(0)}%
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm font-mono">{item.feature}</h4>
                  <p className="text-xs text-muted-foreground mt-0.5">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default FeatureImportanceSection;
