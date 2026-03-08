import { Droplets, MapPin, Database, Layers, ArrowDown, Satellite, Terminal } from "lucide-react";
import { trainingStats } from "@/data/modelData";
import { predictionStats } from "@/data/predictionData";

const HeroSection = () => {
  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <div className="absolute inset-0 bg-background" />
      <div className="absolute inset-0 grid-dots" />
      <div className="absolute inset-0 scanline pointer-events-none" />
      
      <div className="absolute top-1/4 -left-32 w-96 h-96 rounded-full bg-primary/5 blur-3xl" />
      <div className="absolute bottom-1/4 -right-32 w-96 h-96 rounded-full bg-secondary/5 blur-3xl" />

      <div className="relative z-10 container mx-auto px-6 text-center">
        <div className="animate-fade-up">
          <div className="inline-flex items-center gap-2 rounded-md border border-primary/30 bg-primary/5 px-4 py-2 text-sm font-mono text-primary mb-8 glow-primary">
            <Terminal className="h-4 w-4" />
            <span className="text-primary/60">$</span> predict --water-quality --satellite
          </div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-foreground mb-6 leading-tight tracking-tight">
            <span className="text-primary glow-text">Water Quality Prediction</span>
            <br />
            <span className="text-foreground/80 text-3xl md:text-4xl lg:text-5xl font-medium">
              Using Satellite Imagery & ML
            </span>
          </h1>

          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-12 leading-relaxed">
            Fusing <span className="text-secondary font-medium">Landsat imagery</span>, 
            <span className="text-secondary font-medium"> TerraClimate data</span>, and 
            <span className="text-secondary font-medium"> ensemble ML</span> to predict 
            water quality across South Africa's rivers — 
            <span className="text-primary font-semibold font-mono"> best CV R² = 0.87</span>
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 max-w-4xl mx-auto mb-16">
            {[
              { icon: Database, value: trainingStats.samples.toLocaleString(), label: "Training Samples", desc: "Ground-truth measurements" },
              { icon: MapPin, value: predictionStats.uniqueStations.toString(), label: "Test Stations", desc: `${predictionStats.totalPredictions} predictions` },
              { icon: Layers, value: trainingStats.features.toString(), label: "Features", desc: "Engineered inputs" },
              { icon: Satellite, value: "3", label: "Sources", desc: "Satellite + Climate + Ground" },
            ].map((stat) => (
              <div
                key={stat.label}
                className="group rounded-lg border border-border bg-card/50 backdrop-blur-sm p-5 transition-all hover:border-primary/30 hover:glow-primary"
              >
                <stat.icon className="h-5 w-5 mx-auto mb-2 text-primary/70 group-hover:text-primary transition-colors" />
                <div className="text-2xl md:text-3xl font-bold font-mono text-foreground mb-1">{stat.value}</div>
                <div className="text-xs font-medium text-foreground/70">{stat.label}</div>
                <div className="text-xs text-muted-foreground mt-0.5">{stat.desc}</div>
              </div>
            ))}
          </div>
        </div>

        <a
          href="#map"
          className="inline-flex items-center gap-2 text-muted-foreground hover:text-primary transition-colors animate-pulse-gentle"
        >
          <span className="text-sm font-mono">explore_project()</span>
          <ArrowDown className="h-5 w-5" />
        </a>
      </div>
    </section>
  );
};

export default HeroSection;
