import React, { useState } from 'react';
import axios from 'axios';
import { 
  Upload, Brain, ChevronRight, Activity, 
  Database, Zap, Eye
} from 'lucide-react';
import { BarChart, Bar, ResponsiveContainer } from 'recharts';

const CLASS_NAMES = {
  0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
  3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
  6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
  9: "No passing", 10: "No passing for vehicles over 3.5 metric tons",
  11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield",
  14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited",
  17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
  20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
  23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
  26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
  29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
  32: "End of all speed and passing limits", 33: "Turn right ahead",
  34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
  37: "Go straight or left", 38: "Keep right", 39: "Keep left",
  40: "Roundabout mandatory", 41: "End of no passing",
  42: "End of no passing by vehicles over 3.5 metric tons"
};

const App = () => {
  const [history, setHistory] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://127.0.0.1:8000/process', formData);
      const entry = {
        id: Date.now(),
        preview: URL.createObjectURL(file),
        results: res.data.results,
        time: new Date().toLocaleTimeString()
      };
      setHistory([entry, ...history]);
      setCurrentIndex(0);
    } catch (err) {
      alert("API Error: Ensure Python backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const PipelineNode = ({ label, img, sub, active = false }) => (
    <div className={`flex flex-col items-center gap-3 p-4 rounded-2xl transition-all duration-500 ${active ? 'bg-blue-600/10 border border-blue-500/30' : 'bg-zinc-900/50'}`}>
      <span className="text-[10px] font-black tracking-widest text-zinc-500 uppercase">{label}</span>
      <div className="relative group">
        <img src={`data:image/jpeg;base64,${img}`} className={`w-32 h-32 object-cover rounded-xl border-2 shadow-2xl transition-all ${active ? 'border-blue-500 scale-105' : 'border-zinc-800'}`} />
      </div>
      <span className="text-[9px] text-zinc-600 font-medium">{sub}</span>
    </div>
  );

  const activeTest = history[currentIndex];

  return (
    <div className="flex h-screen w-screen bg-[#050506] text-zinc-400 overflow-hidden font-sans m-0 p-0">
      
      {/* LEFT: HISTORY BAR */}
      <aside className="w-20 lg:w-64 border-r border-zinc-900 bg-[#080809] flex flex-col items-center py-6 shrink-0">
        <div className="mb-10 text-blue-500"><Brain size={32} strokeWidth={2.5}/></div>
        
        <label className="mb-8 p-3 bg-blue-600 hover:bg-blue-500 text-white rounded-2xl cursor-pointer transition-transform active:scale-95 shadow-lg shadow-blue-900/40">
          <Upload size={24} />
          <input type="file" className="hidden" onChange={handleUpload} />
        </label>

        <div className="flex-1 w-full px-4 space-y-4 overflow-y-auto no-scrollbar">
          {history.map((h, i) => (
            <div 
              key={h.id} 
              onClick={() => setCurrentIndex(i)}
              className={`relative cursor-pointer group rounded-xl overflow-hidden border-2 transition-all ${i === currentIndex ? 'border-blue-500' : 'border-transparent opacity-50 hover:opacity-100'}`}
            >
              <img src={h.preview} className="w-full aspect-square object-cover" />
            </div>
          ))}
        </div>
      </aside>

      {/* MAIN: FULL WIDTH CONTENT */}
      <main className="flex-1 flex flex-col min-w-0">
        {!activeTest ? (
          <div className="flex-1 flex flex-col items-center justify-center space-y-4">
            <Database size={48} className="text-zinc-800 animate-pulse" />
            <h2 className="text-zinc-600 font-bold uppercase tracking-widest">Neural Link Offline</h2>
          </div>
        ) : (
          <div className="flex-1 w-full p-6 lg:p-10 space-y-12 overflow-y-auto no-scrollbar">
            
            {/* HEADER & RESULT */}
            <div className="w-full flex flex-col lg:flex-row justify-between items-end gap-6">
              <div className="space-y-1">
                <h1 className="text-4xl font-black text-white tracking-tighter">PIPELINE <span className="text-blue-600">INSIGHT</span></h1>
                <p className="text-zinc-500 text-xs">ANALYZING SESSION_{activeTest.id}</p>
              </div>
              <div className="bg-zinc-900/80 backdrop-blur-md border border-zinc-800 p-6 rounded-[2rem] flex items-center gap-6 shadow-2xl">
                <div className="h-12 w-12 bg-green-500/20 rounded-full flex items-center justify-center text-green-500"><Zap /></div>
                <div>
                  <div className="text-[10px] font-bold text-zinc-500 uppercase mb-1">Inference Result</div>
                  <div className="text-2xl font-black text-white leading-none">{CLASS_NAMES[parseInt(activeTest.results[0]?.classification.label)] || "Not Identified"}</div>
                </div>
              </div>
            </div>

            {/* PIPELINE STREAM */}
            {activeTest.results.map((res, idx) => (
              <div key={idx} className="w-full space-y-12 animate-in fade-in slide-in-from-bottom-5 duration-700">
                
                {/* STAGE 1: FULL WIDTH DETECTION */}
                <div className="bg-zinc-900/20 rounded-[3rem] border border-zinc-800/50 p-2 relative group overflow-hidden">
                  <img src={activeTest.preview} className="w-full h-[500px] object-contain rounded-[2.5rem]" />
                  <div className="absolute top-8 left-8 bg-blue-600 text-white text-[10px] font-black px-4 py-2 rounded-full flex items-center gap-2 shadow-xl shadow-blue-900/40 uppercase tracking-widest">
                    <Eye size={14}/> 01 Raw Neural Scan
                  </div>
                </div>

                {/* STAGE 2: THE CHAIN */}
                <div className="bg-[#0c0c0e] rounded-[3.5rem] p-12 border border-zinc-800/40 shadow-2xl">
                  <div className="flex flex-wrap items-center justify-between gap-6">
                    <PipelineNode label="02: Origin" img={res.images.origin_crop} sub="YOLO BBox" />
                    <ChevronRight className="text-zinc-800 shrink-0" size={20} />
                    <PipelineNode label="03: Filtered" img={res.images.after_filter} sub="Processing" active />
                    <ChevronRight className="text-zinc-800 shrink-0" size={20} />
                    <PipelineNode label="04: HOG" img={res.images.hog_viz} sub="Gradient Map" />
                    <ChevronRight className="text-zinc-800 shrink-0" size={20} />
                    <PipelineNode label="05: Hue" img={res.images.hsv_h} sub="Color Space" />
                    <PipelineNode label="06: Sat" img={res.images.hsv_s} sub="Intensity" />
                    <PipelineNode label="07: Val" img={res.images.hsv_v} sub="Luminance" />
                  </div>
                </div>

                {/* STAGE 3: DATA VISUALS */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 pb-12">
                  <div className="bg-zinc-900/40 p-8 rounded-[3rem] border border-zinc-800/50">
                    <div className="flex items-center gap-2 mb-8"><Activity size={16} className="text-indigo-500"/> <span className="text-[10px] font-black uppercase tracking-widest text-zinc-500">Mathematical Tensor Sample</span></div>
                    <div className="h-40">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={res.feature_sample.map(v => ({ v }))}>
                          <Bar dataKey="v" fill="#6366f1" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-blue-600 to-indigo-800 p-10 rounded-[3rem] flex flex-col justify-center text-white shadow-2xl shadow-blue-900/30">
                    <div className="text-[10px] font-black uppercase tracking-[0.3em] mb-6 opacity-60">XGBoost Decision Engine</div>
                    <div className="text-4xl font-black leading-tight mb-6">{CLASS_NAMES[parseInt(res.classification.label)]}</div>
                    <div className="flex items-center gap-6">
                      <div className="flex-1 h-4 bg-black/20 rounded-full overflow-hidden border border-white/10">
                        <div className="h-full bg-white shadow-[0_0_15px_rgba(255,255,255,0.5)] transition-all duration-1000" style={{width: `${res.classification.confidence * 100}%`}} />
                      </div>
                      <span className="font-mono text-xl font-black">{(res.classification.confidence * 100).toFixed(2)}%</span>
                    </div>
                  </div>
                </div>

              </div>
            ))}
          </div>
        )}
      </main>

      {/* LOADING OVERLAY */}
      {loading && (
        <div className="fixed inset-0 bg-[#050506]/95 backdrop-blur-2xl z-50 flex flex-col items-center justify-center">
          <div className="w-24 h-24 border-2 border-blue-500/10 border-t-blue-500 rounded-full animate-spin mb-10" />
          <div className="text-xs font-black text-white tracking-[0.5em] animate-pulse uppercase">Decomposing Neural Tensors</div>
        </div>
      )}
    </div>
  );
};

export default App;