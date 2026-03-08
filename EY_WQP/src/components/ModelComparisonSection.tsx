import { modelComparisonData } from "@/data/modelData";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Trophy } from "lucide-react";

const getBestModel = (key: "ta" | "ec" | "drp") => {
  return modelComparisonData.reduce((best, m) => (m[key] > best[key] ? m : best));
};

const ModelComparisonSection = () => {
  const bestTA = getBestModel("ta");
  const bestEC = getBestModel("ec");
  const bestDRP = getBestModel("drp");

  return (
    <section id="models" className="py-24 bg-background relative">
      <div className="absolute inset-0 grid-dots opacity-30" />
      <div className="container mx-auto px-6 relative z-10">
        <div className="flex items-center gap-2 mb-6">
          <span className="text-xs font-mono text-primary/60">05</span>
          <div className="h-px flex-1 bg-border" />
          <span className="text-xs font-mono text-primary uppercase tracking-widest">Model Selection</span>
          <div className="h-px flex-1 bg-border" />
        </div>

        <div className="text-center mb-12">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
            12 Candidates, Best Per Target
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-sm">
            Rigorous 5-fold CV across model families. No single model wins everywhere.
          </p>
        </div>

        {/* Winners */}
        <div className="grid md:grid-cols-3 gap-3 mb-10">
          {[
            { label: "Total Alkalinity", model: bestTA.fullName, r2: bestTA.ta },
            { label: "Elec. Conductance", model: bestEC.fullName, r2: bestEC.ec },
            { label: "Dissolved Reactive P", model: bestDRP.fullName, r2: bestDRP.drp },
          ].map((item) => (
            <div key={item.label} className="rounded-lg border border-primary/20 bg-primary/5 p-5 text-center glow-primary">
              <Trophy className="h-5 w-5 text-primary mx-auto mb-2" />
              <h4 className="font-medium text-foreground text-sm mb-1">{item.label}</h4>
              <p className="text-2xl font-bold font-mono text-primary mb-1">{item.r2.toFixed(4)}</p>
              <p className="text-xs text-muted-foreground font-mono">{item.model}</p>
            </div>
          ))}
        </div>

        {/* Table */}
        <Card className="bg-card border-border overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-mono text-foreground">model_comparison(metric="cv_r2")</CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/30">
                  <th className="text-left py-3 px-4 font-mono text-xs text-muted-foreground uppercase tracking-wider">Model</th>
                  <th className="text-center py-3 px-4 font-mono text-xs text-muted-foreground uppercase tracking-wider">TA</th>
                  <th className="text-center py-3 px-4 font-mono text-xs text-muted-foreground uppercase tracking-wider">EC</th>
                  <th className="text-center py-3 px-4 font-mono text-xs text-muted-foreground uppercase tracking-wider">DRP</th>
                </tr>
              </thead>
              <tbody>
                {modelComparisonData.map((m, i) => {
                  const isBestTA = m.name === bestTA.name;
                  const isBestEC = m.name === bestEC.name;
                  const isBestDRP = m.name === bestDRP.name;
                  return (
                    <tr key={m.name} className={`border-b border-border/30 ${i % 2 === 0 ? "bg-muted/10" : ""} hover:bg-muted/20 transition-colors`}>
                      <td className="py-2.5 px-4 font-mono text-xs text-foreground">{m.fullName}</td>
                      <td className={`py-2.5 px-4 text-center font-mono text-xs ${isBestTA ? "text-primary font-bold" : "text-muted-foreground"}`}>
                        {m.ta.toFixed(4)} {isBestTA && "🏆"}
                      </td>
                      <td className={`py-2.5 px-4 text-center font-mono text-xs ${isBestEC ? "text-primary font-bold" : "text-muted-foreground"}`}>
                        {m.ec.toFixed(4)} {isBestEC && "🏆"}
                      </td>
                      <td className={`py-2.5 px-4 text-center font-mono text-xs ${isBestDRP ? "text-primary font-bold" : "text-muted-foreground"}`}>
                        {m.drp.toFixed(4)} {isBestDRP && "🏆"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default ModelComparisonSection;
