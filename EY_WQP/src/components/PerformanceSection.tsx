import { bestModels, trainingStats } from "@/data/modelData";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const MetricGauge = ({ label, value, model }: { label: string; value: number; model: string }) => {
  const percentage = value * 100;
  return (
    <div className="rounded-lg border border-border bg-card p-6 text-center hover:border-primary/20 transition-all">
      <div className="relative w-24 h-24 mx-auto mb-4">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
          <path
            d="M18 2.0845a15.9155 15.9155 0 0 1 0 31.831a15.9155 15.9155 0 0 1 0-31.831"
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth="2.5"
          />
          <path
            d="M18 2.0845a15.9155 15.9155 0 0 1 0 31.831a15.9155 15.9155 0 0 1 0-31.831"
            fill="none"
            stroke="hsl(var(--primary))"
            strokeWidth="2.5"
            strokeDasharray={`${percentage}, 100`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xl font-bold font-mono text-foreground">{value}</span>
        </div>
      </div>
      <h4 className="font-semibold text-foreground text-sm">{label}</h4>
      <p className="text-xs text-muted-foreground font-mono mt-1">{model}</p>
      <p className="text-xs text-primary font-mono mt-0.5">CV R²</p>
    </div>
  );
};

const PerformanceSection = () => {
  return (
    <section id="performance" className="py-24 bg-card/50 relative">
      <div className="absolute inset-0 grid-dots opacity-30" />
      <div className="container mx-auto px-6 relative z-10">
        <div className="flex items-center gap-2 mb-6">
          <span className="text-xs font-mono text-primary/60">04</span>
          <div className="h-px flex-1 bg-border" />
          <span className="text-xs font-mono text-primary uppercase tracking-widest">Model Performance</span>
          <div className="h-px flex-1 bg-border" />
        </div>

        <div className="text-center mb-12">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
            Evaluation Results
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-sm">
            5-fold cross-validation on {trainingStats.samples.toLocaleString()} samples.
            Best model selected per target from 12 candidates.
          </p>
        </div>

        {/* R² Gauges */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <MetricGauge label="Total Alkalinity" value={bestModels.ta.cvR2} model={bestModels.ta.model} />
          <MetricGauge label="Electrical Conductance" value={bestModels.ec.cvR2} model={bestModels.ec.model} />
          <MetricGauge label="Dissolved Reactive Phosphorus" value={bestModels.drp.cvR2} model={bestModels.drp.model} />
        </div>

        {/* Training stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: "Training Samples", value: trainingStats.samples.toLocaleString() },
            { label: "Features", value: trainingStats.features.toString() },
            { label: "CV Folds", value: "5" },
            { label: "Model Candidates", value: "12" },
          ].map((stat) => (
            <div key={stat.label} className="rounded-lg border border-border bg-card p-4 text-center">
              <div className="text-lg font-bold font-mono text-primary">{stat.value}</div>
              <div className="text-xs text-muted-foreground mt-1">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default PerformanceSection;
