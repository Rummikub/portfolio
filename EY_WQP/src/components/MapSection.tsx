import { useState, useEffect, useRef } from "react";
import { uniqueLocations, predictionStats, type UniqueLocation } from "@/data/predictionData";
import { Card, CardContent } from "@/components/ui/card";
import { X } from "lucide-react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const MapSection = () => {
  const [selected, setSelected] = useState<UniqueLocation | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<L.CircleMarker[]>([]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [-32.8, 26.5],
      zoom: 6,
      scrollWheelZoom: true,
    });

    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      attribution: "Tiles &copy; Esri",
    }).addTo(map);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    }).addTo(map);

    uniqueLocations.forEach((loc) => {
      const marker = L.circleMarker([loc.lat, loc.lng], {
        radius: Math.max(5, Math.min(12, loc.sampleCount * 1.5)),
        fillColor: "hsl(168, 80%, 46%)",
        color: "rgba(255,255,255,0.6)",
        weight: 2,
        fillOpacity: 0.8,
      }).addTo(map);

      marker.on("click", () => {
        setSelected(loc);
      });

      markersRef.current.push(marker);
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
      markersRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (selected && mapRef.current) {
      mapRef.current.flyTo([selected.lat, selected.lng], 8, { duration: 1 });
    }
  }, [selected]);

  return (
    <section id="map" className="py-24 bg-card/50 relative">
      <div className="absolute inset-0 grid-dots opacity-30" />
      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-12">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
            Prediction Locations
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-sm">
            {predictionStats.totalPredictions} predictions across {predictionStats.uniqueStations} monitoring stations in South Africa.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="rounded-lg border border-border overflow-hidden">
              <div
                ref={containerRef}
                className="aspect-[4/3]"
                style={{ zIndex: 0 }}
              />
              <div className="flex gap-4 px-4 py-3 text-xs font-mono bg-card border-t border-border">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-primary" /> Prediction Site
                </span>
                <span className="text-muted-foreground">
                  Marker size = sample count
                </span>
              </div>
            </div>
          </div>

          <div>
            {selected ? (
              <div className="sticky top-24 rounded-lg border border-border bg-card p-5">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="font-bold text-foreground font-mono text-sm">
                      {selected.lat.toFixed(2)}°S, {selected.lng.toFixed(2)}°E
                    </h3>
                    <p className="text-xs text-muted-foreground font-mono">{selected.sampleCount} prediction{selected.sampleCount > 1 ? "s" : ""}</p>
                  </div>
                  <button onClick={() => setSelected(null)} className="text-muted-foreground hover:text-foreground">
                    <X className="h-4 w-4" />
                  </button>
                </div>

                <div className="space-y-4">
                  {[
                    { label: "Avg. Total Alkalinity", value: `${selected.avgAlkalinity} mg/L`, max: predictionStats.alkalinity.max, current: selected.avgAlkalinity },
                    { label: "Avg. Elec. Conductance", value: `${selected.avgConductance} mS/m`, max: predictionStats.conductance.max, current: selected.avgConductance },
                    { label: "Avg. DRP", value: `${selected.avgPhosphorus} µg/L`, max: predictionStats.phosphorus.max, current: selected.avgPhosphorus },
                  ].map((param) => (
                    <div key={param.label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-muted-foreground">{param.label}</span>
                        <span className="font-mono font-medium text-foreground">{param.value}</span>
                      </div>
                      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full transition-all duration-500"
                          style={{ width: `${Math.min((param.current / param.max) * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="sticky top-24 rounded-lg border border-border/50 bg-card p-6 text-center">
                <p className="text-muted-foreground text-sm font-mono">click a marker to inspect</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default MapSection;
