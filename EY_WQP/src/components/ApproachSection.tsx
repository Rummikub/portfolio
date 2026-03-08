import { Satellite, CloudRain, Droplets, ChevronRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const dataSources = [
  {
    name: "Water Quality Measurements",
    description: "Ground-truth water samples collected from monitoring stations across South Africa between 2011-2015.",
    features: ["Latitude", "Longitude", "Sample Date", "Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"],
    icon: Droplets,
  },
  {
    name: "Landsat Satellite Imagery",
    description: "Spectral band data from NASA's Landsat satellites, capturing how water and surrounding land reflect light at different wavelengths.",
    features: ["NIR (Near Infrared)", "Green Band", "SWIR16", "SWIR22", "NDMI", "MNDWI"],
    icon: Satellite,
  },
  {
    name: "TerraClimate Data",
    description: "Monthly climate variables from the TerraClimate dataset, providing information about evapotranspiration and water balance.",
    features: ["PET (Potential Evapotranspiration)"],
    icon: CloudRain,
  },
];

const engineeredFeatures = [
  { category: "Spectral Indices", count: 3, examples: ["NDWI", "NDMI", "MNDWI"], description: "Water and moisture indices from satellite bands" },
  { category: "Band Ratios", count: 3, examples: ["NIR/SWIR16", "Green/SWIR22"], description: "Ratios between spectral bands reveal surface properties" },
  { category: "Climate Interactions", count: 3, examples: ["PET × NDMI", "PET × MNDWI"], description: "How climate interacts with spectral signals" },
  { category: "Log Transforms", count: 5, examples: ["log(NIR)", "log(Green)"], description: "Compress skewed distributions for better modeling" },
  { category: "Squared Terms", count: 3, examples: ["NDMI²", "MNDWI²"], description: "Capture non-linear relationships" },
  { category: "Temporal Encoding", count: 5, examples: ["month_sin", "month_cos"], description: "Cyclical time features for seasonal patterns" },
  { category: "Geographic", count: 2, examples: ["Latitude", "Longitude"], description: "Spatial coordinates as features" },
  { category: "Raw Bands", count: 5, examples: ["NIR", "Green", "SWIR16"], description: "Original satellite and climate measurements" },
];

const ApproachSection = () => {
  return (
    <section id="approach" className="py-24 bg-background relative">
      <div className="absolute inset-0 grid-dots opacity-50" />
      <div className="container mx-auto px-6 relative z-10">
        {/* Problem Statement */}
        <div className="max-w-3xl mx-auto mb-24">
          <div className="flex items-center gap-2 mb-6">
            <span className="text-xs font-mono text-primary/60">01</span>
            <div className="h-px flex-1 bg-border" />
            <span className="text-xs font-mono text-primary uppercase tracking-widest">Problem Statement</span>
            <div className="h-px flex-1 bg-border" />
          </div>
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6 text-center">
            Why Predict Water Quality <span className="text-primary">from Space</span>?
          </h2>
          <div className="space-y-4 text-muted-foreground leading-relaxed">
            <p>
              Traditional water quality monitoring requires <span className="text-foreground font-medium">physical sample collection</span> at each site — 
              expensive, time-consuming, and limited in spatial coverage. South Africa has ~200 monitoring stations, 
              but thousands of kilometers of rivers remain unmonitored.
            </p>
            <p>
              <span className="text-primary font-medium">Satellite remote sensing</span> offers continuous, large-scale observations. 
              By training ML models on paired ground-truth measurements and satellite data, we can predict water quality parameters 
              at <span className="text-foreground font-medium">any location visible from space</span>.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-3 mt-8">
            {["Total Alkalinity (mg/L)", "Electrical Conductance (mS/m)", "Dissolved Reactive Phosphorus (µg/L)"].map((target) => (
              <div key={target} className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-center">
                <span className="text-xs font-mono text-primary">{target}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Data Understanding */}
        <div className="mb-24">
          <div className="flex items-center gap-2 mb-6">
            <span className="text-xs font-mono text-primary/60">02</span>
            <div className="h-px flex-1 bg-border" />
            <span className="text-xs font-mono text-primary uppercase tracking-widest">Data Understanding</span>
            <div className="h-px flex-1 bg-border" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4 text-center">
            Three Data Sources, One Model
          </h2>
          <p className="text-muted-foreground text-center max-w-2xl mx-auto mb-10">
            Ground-truth water measurements paired with satellite imagery and climate data spanning 2011–2015.
          </p>

          <div className="grid md:grid-cols-3 gap-4">
            {dataSources.map((source) => {
              const Icon = source.icon;
              return (
                <Card key={source.name} className="bg-card border-border hover:border-primary/30 transition-all group">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center border border-primary/20">
                        <Icon className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-foreground text-sm">{source.name}</h3>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed mb-4">{source.description}</p>
                    <div className="flex flex-wrap gap-1">
                      {source.features.map((f) => (
                        <span key={f} className="text-xs font-mono px-2 py-0.5 rounded bg-muted text-muted-foreground border border-border">
                          {f}
                        </span>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Feature Engineering */}
        <div>
          <div className="flex items-center gap-2 mb-6">
            <span className="text-xs font-mono text-primary/60">03</span>
            <div className="h-px flex-1 bg-border" />
            <span className="text-xs font-mono text-primary uppercase tracking-widest">Feature Engineering</span>
            <div className="h-px flex-1 bg-border" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4 text-center">
            29 Features from Raw Data
          </h2>
          <p className="text-muted-foreground text-center max-w-2xl mx-auto mb-10">
            Transforming satellite bands and climate variables into meaningful ML inputs.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {engineeredFeatures.map((cat) => (
              <div key={cat.category} className="rounded-lg border border-border bg-card p-4 hover:border-primary/20 transition-all">
                <div className="text-2xl font-bold font-mono text-primary mb-1">{cat.count}</div>
                <h4 className="font-semibold text-foreground text-sm mb-1">{cat.category}</h4>
                <p className="text-xs text-muted-foreground mb-3">{cat.description}</p>
                <div className="flex flex-wrap gap-1">
                  {cat.examples.slice(0, 2).map((e) => (
                    <span key={e} className="text-xs font-mono px-1.5 py-0.5 rounded bg-muted text-primary/70 border border-border">{e}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Pipeline visualization */}
          <div className="mt-16 flex flex-col md:flex-row items-center justify-center gap-2">
            {[
              { label: "Raw Data", sub: "3 sources" },
              { label: "Clean & Merge", sub: "paired samples" },
              { label: "Engineer", sub: "29 features" },
              { label: "Train & Validate", sub: "ensemble models" },
              { label: "Predict", sub: "200 test samples" },
            ].map((step, i, arr) => (
              <div key={step.label} className="flex items-center gap-2">
                <div className="text-center px-4 py-3 rounded-lg border border-border bg-card min-w-[120px]">
                  <div className="font-semibold text-foreground text-xs">{step.label}</div>
                  <div className="text-xs text-muted-foreground font-mono mt-0.5">{step.sub}</div>
                </div>
                {i < arr.length - 1 && (
                  <ChevronRight className="h-4 w-4 text-primary/40 flex-shrink-0 hidden md:block" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ApproachSection;
