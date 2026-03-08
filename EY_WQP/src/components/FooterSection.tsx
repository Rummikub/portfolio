import { Satellite } from "lucide-react";

const FooterSection = () => {
  return (
    <footer className="bg-card border-t border-border py-16">
      <div className="container mx-auto px-6">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Satellite className="h-5 w-5 text-primary" />
              <span className="text-lg font-bold font-mono text-foreground">Water Quality Prediction</span>
            </div>
            <p className="text-muted-foreground max-w-md leading-relaxed text-sm">
              An ML research project predicting water quality from satellite imagery across South Africa's river systems. Built for the EY Open Science Data Challenge 2026.
            </p>
            <div className="flex gap-2 mt-4">
              <span className="text-xs font-mono px-2 py-1 rounded bg-accent text-accent-foreground">Python</span>
              <span className="text-xs font-mono px-2 py-1 rounded bg-accent text-accent-foreground">XGBoost</span>
              <span className="text-xs font-mono px-2 py-1 rounded bg-accent text-accent-foreground">Landsat</span>
              <span className="text-xs font-mono px-2 py-1 rounded bg-accent text-accent-foreground">React</span>
            </div>
          </div>

          <div className="flex flex-col items-start md:items-end gap-4">
            <h4 className="font-medium text-foreground/80 text-sm font-mono">// research project</h4>
            <p className="text-sm text-muted-foreground text-right max-w-xs">
              Built for the EY Open Science Data Challenge 2026 — Optimizing Clean Water Supply.
            </p>
          </div>
        </div>

        <div className="border-t border-border mt-12 pt-6 text-center text-xs text-muted-foreground font-mono">
          <span className="text-primary/40">{'>'}</span> Built with React • Recharts • Leaflet • EY Data Challenge 2026
        </div>
      </div>
    </footer>
  );
};

export default FooterSection;