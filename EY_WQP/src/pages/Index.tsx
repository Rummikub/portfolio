import { Satellite } from "lucide-react";
import HeroSection from "@/components/HeroSection";
import MapSection from "@/components/MapSection";
import ApproachSection from "@/components/ApproachSection";
import PerformanceSection from "@/components/PerformanceSection";
import ModelComparisonSection from "@/components/ModelComparisonSection";
import FeatureImportanceSection from "@/components/FeatureImportanceSection";
import PredictionChartsSection from "@/components/PredictionChartsSection";
import FooterSection from "@/components/FooterSection";

const navLinks = [
  { href: "#map", label: "Data" },
  { href: "#approach", label: "Pipeline" },
  { href: "#performance", label: "Results" },
  { href: "#models", label: "Models" },
  { href: "#features", label: "Features" },
  { href: "#predictions", label: "Predictions" },
];

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <nav className="fixed top-4 left-1/2 -translate-x-1/2 z-50 bg-card/80 backdrop-blur-lg rounded-lg px-5 py-2.5 shadow-lg border border-border/50">
        <div className="flex items-center gap-5">
          <a href="#hero" className="flex items-center gap-1.5 text-primary font-bold text-sm font-mono">
            <Satellite className="h-4 w-4" />
            <span className="hidden sm:inline">WQP</span>
          </a>
          <div className="w-px h-4 bg-border" />
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="text-xs font-mono text-muted-foreground hover:text-primary transition-colors hidden md:block"
            >
              {link.label}
            </a>
          ))}
        </div>
      </nav>

      <HeroSection />
      <MapSection />
      <ApproachSection />
      <PerformanceSection />
      <ModelComparisonSection />
      <FeatureImportanceSection />
      <PredictionChartsSection />
      <FooterSection />
    </div>
  );
};

export default Index;
