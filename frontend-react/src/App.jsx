import React, { useState, useRef } from "react";

// Ensure this matches your backend URL
const BACKEND = "http://127.0.0.1:8000"; 
const STREAM_ENDPOINT = `${BACKEND}/predict_stream`;

// --- NEW: Define all model choices ---
const MODEL_OPTIONS = [
    "CNN15", 
    "ResNet50", 
    "VGG16", 
    "DenseNet121", 
    "EfficientNet121", 
    "ViT", 
    "CNN10", 
    "CNN20"
];

export default function App() {
  const [file, setFile] = useState(null);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState("");
  const [result, setResult] = useState(null);
  // --- NEW STATE: Model Selection ---
  const [modelChoice, setModelChoice] = useState(MODEL_OPTIONS[0]); 
  const logRef = useRef();

  // Helper to scroll logs to bottom
  const append = (line) => {
    setLogs((prev) => {
      const next = [...prev, line];
      setTimeout(() => {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
      }, 20);
      return next;
    });
  };

  const onFile = (e) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setLogs([]);
    setProgress("");
  };

  const startStream = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }
    setRunning(true);
    setLogs([]);
    setResult(null);
    setProgress("Starting upload...");

    const form = new FormData();
    form.append("file", file);
    // --- NEW: Append the model choice to the form data ---
    form.append("model_choice", modelChoice); 

    try {
      const resp = await fetch(STREAM_ENDPOINT, {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        const txt = await resp.text();
        append(`Server error ${resp.status}: ${txt}`);
        setRunning(false);
        return;
      }

      append("Connection established. Processing...");
      
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let done = false;
      let buffer = "";

      while (!done) {
        const { value, done: d } = await reader.read();
        done = d;
        if (value) {
          buffer += dec.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop(); 

          for (let line of lines) {
            if (!line.trim()) continue;
            
            let cleanLine = line.replace(/^data:\s*/, "").trim();

            const jsonStart = cleanLine.indexOf('{');
            const jsonEnd = cleanLine.lastIndexOf('}');
            
            if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
                const maybeJson = cleanLine.substring(jsonStart, jsonEnd + 1);
                
                try {
                    const parsed = JSON.parse(maybeJson);
                    if (parsed.final_label && parsed.per_class) {
                        setResult(parsed);
                        setProgress("Analysis Complete.");
                        append("Final result received.");
                        continue; 
                    }
                } catch (e) {
                    // Not valid final JSON, or parsing failed. Fall through to print as log.
                }
            }

            append(cleanLine);
            setProgress(cleanLine);
          }
        }
      }
      
    } catch (err) {
      append("Network error: " + err.message);
    } finally {
      setRunning(false);
    }
  };


  return (
    <div className="app-root">
      <header className="header">
        <div>
          <h1 className="title">Parkinson MRI Predictor</h1>
          <p className="subtitle">
            Upload a brain MRI (.nii .nii.gz or zipped DICOM) to detect Parkinson's Disease.
          </p>
        </div>
      </header>

      <div className="grid md:grid-cols-2 gap-6">
        {/* LEFT PANEL: INPUT & RESULTS */}
        <div className="panel">
          
          {/* --- NEW: Model Selection Dropdown --- */}
          <label className="block text-sm font-medium mb-1 text-slate-700">Model Selection</label>
          <select 
            value={modelChoice} 
            onChange={(e) => setModelChoice(e.target.value)}
            className="block w-full px-3 py-2 border border-slate-300 rounded-lg shadow-sm text-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 mb-4"
            disabled={running}
          >
            {MODEL_OPTIONS.map(model => (
                <option key={model} value={model}>
                    {model} 
                </option>
            ))}
          </select>
          
          <label className="block text-sm font-medium mb-2 text-slate-700">MRI Volume</label>
          <input 
            type="file" 
            onChange={onFile} 
            accept=".nii,.nii.gz,.zip,.dcm" 
            className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-brand-50 file:text-brand-700 hover:file:bg-brand-100 mb-4"
          />
          
          <div className="flex gap-3 mb-6">
            <button 
              className={`btn btn-primary ${running ? 'opacity-50 cursor-not-allowed' : ''}`}
              onClick={startStream} 
              disabled={running}
            >
              {running ? "Analyzing..." : "Upload & Analyze"}
            </button>
            
            <button 
              className="btn btn-ghost" 
              onClick={() => { setFile(null); setLogs([]); setResult(null); }}
            >
              Reset
            </button>
          </div>

          {/* RESULT CARD */}
          <div className="result-card">
             <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Prediction Result ({modelChoice})</div>
             
             {result ? (
               <div>
                 <div className="flex items-center justify-between mb-4">
                   <div className="text-3xl font-bold capitalize text-slate-800">
                     {result.final_label}
                   </div>
                   <div className="text-sm font-medium bg-white px-3 py-1 rounded-full shadow-sm border">
                     Confidence: {(result.final_confidence * 100).toFixed(1)}%
                   </div>
                 </div>

                 {/* BAR CHART */}
                 {result.per_class && (
                   <div className="space-y-3">
                     {Object.entries(result.per_class).map(([label, score]) => (
                       <div key={label}>
                         <div className="flex justify-between text-xs font-medium text-slate-600 mb-1 uppercase">
                           <span>{label}</span>
                           <span>{(score * 100).toFixed(1)}%</span>
                         </div>
                         <div className="conf-bar">
                           <div 
                              className={`h-full rounded transition-all duration-500 ${
                                label === result.final_label ? 'bg-brand-600' : 'bg-slate-300'
                              }`}
                              style={{ width: `${score * 100}%` }} 
                           />
                         </div>
                       </div>
                     ))}
                   </div>
                 )}
               </div>
             ) : (
               <div className="text-sm text-slate-500 italic">
                 Results will appear here after analysis.
               </div>
             )}
          </div>
        </div>

        {/* RIGHT PANEL: LOGS */}
        <div className="panel flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-bold text-slate-700">Processing Logs</div>
            <div className="text-xs text-slate-400">
               {running ? <span className="text-brand-600 animate-pulse">● Live</span> : "Idle"}
            </div>
          </div>

          <div className="log-box flex-1" ref={logRef}>
            {logs.length === 0 ? (
              <div className="text-slate-400 italic mt-10 text-center">
                Ready to process...
              </div>
            ) : (
              logs.map((L, idx) => (
                <div key={idx} className="py-1 border-b border-slate-100 last:border-0">
                  <span className="text-slate-400 mr-2 text-xs">[{idx + 1}]</span>
                  {L}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      
      <div className="text-center mt-8 text-xs text-slate-400">
        Parkinson Detection BTP • Research Use Only
      </div>
    </div>
  );
}