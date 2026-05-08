"""
LLM.py  v7  —  AI Scarecrow RL Edition
========================================
MKL-DNN fix + correct detection thresholds to stop elephant false positives.
Delegates all vision to detect.py (YOLO detect + MOG2 motion + ResNet classify).
"""

# ── MKL-DNN FIX (must be before any torch import) ────────────────────────────
import os
os.environ["PYTORCH_JIT"]           = "0"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["OMP_NUM_THREADS"]        = "1"

import torch
torch.backends.mkldnn.enabled = False

import time, json, threading, tempfile, wave, random, shutil, math, warnings
import requests, cv2, pygame, numpy as np
import torch.nn as nn
import detect as detector
from groq import Groq
from torchvision import models
from collections import defaultdict
from ultralytics import YOLO as YOLOModel

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_KEY    = os.getenv("GROQ_API_KEY",      "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")
FS_KEY      = os.getenv("FREESOUND_API_KEY",  "7au49c6sOxqaEv3zdmEc6szarXTUlktlD2QyXOi9")
RESNET_PATH = detector.RESNET_PATH
YOLO_MODEL  = detector.YOLO_MODEL
CLASSES     = detector.CLASSES
ANIMALS     = [c for c in CLASSES if c != 'no animal']
DEVICE      = detector.DEVICE
RL_FILE     = "scarecrow_rl.json"
CACHE_DIR   = "scarecrow_sound_cache"

# ── TIMING ────────────────────────────────────────────────────────────────────
TRIG_FRAMES    = 5      # consecutive confirmations before trigger (was 3-4 → too easy)
COOLDOWN       = 20
MIN_PLAY_S     = 25     # always plays at least 25 seconds
MAX_PLAY       = 90
STOP_ABSENT    = 10     # must be absent 10s AFTER min play
CLASSIFY_N     = 2
CONFIRM_HOLD_S = 3.0

# ── RL ────────────────────────────────────────────────────────────────────────
EFF_THRESH = 0.35
MIN_TRIALS = 3
TIER_T2    = 4
TIER_T3    = 8
UCB_C      = 1.0

groq_client = Groq(api_key=GROQ_KEY)

# ── SOUND DEFINITIONS ─────────────────────────────────────────────────────────
SOUNDS = {
    "Peacock": {
        1: {"q": ["hawk attack screech",     "eagle hunting scream",     "peregrine falcon attack"],
            "ban": ["music","cartoon","kids","synth","ambient","indoor","loop","soft"]},
        2: {"q": ["barn owl screech attack",  "great horned owl screech", "owl attack sound"],
            "ban": ["music","cartoon","kids","indoor","loop","ambient"]},
        3: {"q": ["raptor hunting screech",   "eagle owl attack call",    "predator bird attack"],
            "ban": ["music","cartoon","kids","indoor"]},
    },
    "bonnet_macaque": {
        1: {"q": ["leopard growl snarl",      "leopard attack sound",     "big cat snarl close"],
            "ban": ["music","cartoon","kids","pet","purring","domestic","indoor","soft"]},
        2: {"q": ["tiger growl close",        "jaguar growl attack",      "lion growl loud"],
            "ban": ["music","cartoon","kids","crowd","indoor","distant"]},
        3: {"q": ["large predator growl",     "wild cat attack sound",    "predator snarl loud"],
            "ban": ["music","cartoon","kids","indoor","soft"]},
    },
    "chital": {
        1: {"q": ["tiger roar close loud",    "Bengal tiger roar",        "tiger roar jungle"],
            "ban": ["music","cartoon","kids","crowd","stadium","indoor","distant","soft"]},
        2: {"q": ["leopard growl hunt",       "big cat hunting call",     "wild predator roar"],
            "ban": ["music","cartoon","kids","indoor","distant"]},
        3: {"q": ["wolf pack howl attack",    "wolf growl menacing",      "wolves hunting night"],
            "ban": ["music","cartoon","kids","dog","domestic","indoor","soft"]},
    },
    "elephant": {
        1: {"q": ["honey bee swarm attack",   "bee hive swarm angry",     "angry bee colony buzzing"],
            "ban": ["music","cartoon","kids","electric","synth","funny","indoor","loop","soft"]},
        2: {"q": ["wasp swarm attack",        "hornet nest disturbed",    "bee sting alarm swarm"],
            "ban": ["music","cartoon","kids","electric","synth","indoor","soft"]},
        3: {"q": ["explosion blast outdoor",  "dynamite explosion loud",  "loud explosion field"],
            "ban": ["music","kids","indoor","distant","soft","echo"]},
    },
    "pig": {
        1: {"q": ["wolf howling pack",        "wolf growl menacing",      "wolves hunting howl"],
            "ban": ["music","cartoon","kids","dog","pet","domestic","indoor","soft"]},
        2: {"q": ["wild wolf snarl attack",   "wolf hunt growl",          "wolf attack sound"],
            "ban": ["music","cartoon","kids","domestic","indoor","soft"]},
        3: {"q": ["bear growl attack loud",   "large predator approach",  "bear snarl close"],
            "ban": ["music","cartoon","kids","indoor","distant","soft"]},
    },
    "porcupine": {
        1: {"q": ["leopard snarl attack",     "big cat attack growl",     "leopard hunting sound"],
            "ban": ["music","cartoon","kids","pet","domestic","purr","indoor","soft"]},
        2: {"q": ["eagle owl screech attack", "large owl hunting call",   "owl attack screech loud"],
            "ban": ["music","cartoon","kids","indoor","distant","soft"]},
        3: {"q": ["wild cat hissing growl",   "predator attack snarl",    "big cat hiss attack"],
            "ban": ["music","cartoon","kids","indoor","soft"]},
    },
    "street_dogs": {
        1: {"q": ["angry man shouting loud",  "aggressive human yelling", "man authority shout"],
            "ban": ["music","song","kids","crowd","ambient","ringtone",
                    "speech","podcast","laugh","background","indoor","loop"]},
        2: {"q": ["loud firecracker bang",    "single firecracker burst", "firecracker outdoor"],
            "ban": ["music","kids","indoor","ambient","distant","soft"]},
        3: {"q": ["air horn loud blast",      "foghorn boat horn",        "loud horn blast outdoor"],
            "ban": ["music","cartoon","kids","indoor","soft","distant"]},
    },
    "wild_boar": {
        1: {"q": ["rifle gunshot outdoor",    "hunting rifle shot forest","single rifle shot loud"],
            "ban": ["music","cartoon","kids","indoor","silenced","pop","soft","distant"]},
        2: {"q": ["shotgun blast outdoor",    "12 gauge shotgun shot",    "shotgun single shot"],
            "ban": ["music","cartoon","kids","indoor","soft","distant"]},
        3: {"q": ["air raid siren wailing",   "emergency siren loud",     "loud alarm siren outdoor"],
            "ban": ["music","kids","indoor","soft","distant","muffled"]},
    },
}

SCIENCE = {
    "Peacock":        "Raptor calls trigger hardwired scatter reflex in peafowl",
    "bonnet_macaque": "Leopard growl activates colony alarm chain → instant group flight",
    "chital":         "Tiger roar causes cortisol spike and instant flight in deer",
    "elephant":       "Honeybee swarm causes panic — bees attack trunk and eyes (King 2007)",
    "pig":            "Wolf howl triggers ancient hardwired predator-flee response",
    "porcupine":      "Leopard snarl triggers spine-raising defensive retreat",
    "street_dogs":    "Angry human voice triggers learned submission and flight",
    "wild_boar":      "Gunshot activates centuries of hunting-pressure flight reflex",
}

# ── LOGGING ───────────────────────────────────────────────────────────────────
_plk = threading.Lock()
def log(tag, msg):
    with _plk: print(f"[{time.strftime('%H:%M:%S')}][{tag:<9}] {msg}", flush=True)
def section(t):
    with _plk: print(f"\n{'='*60}\n  {t}\n{'='*60}", flush=True)

# ── RL ────────────────────────────────────────────────────────────────────────
def _rk(a,t,q):  return f"{a}::{t}::{q}"
def _avg(r):      return round(sum(r["scores"])/len(r["scores"]),3) if r["scores"] else 0.5
def get_tier(v):  return 3 if v>=TIER_T3 else 2 if v>=TIER_T2 else 1

def get_rec(m, animal, tier, query):
    k = _rk(animal, tier, query)
    if k not in m["sounds"]:
        m["sounds"][k] = {"uses":0,"scores":[],"q":0.5,"blacklisted":False}
    return m["sounds"][k]

def load_rl():
    m = {"sounds":{},"events":[],"visits":{}}
    if not os.path.exists(RL_FILE):
        log("RL","No prior data"); return m
    try:
        d = json.load(open(RL_FILE, encoding="utf-8"))
        events = [e for e in d.get("events",[])
                  if {"animal","query","tier","eff"} <= e.keys()]
        for ev in events:
            r = get_rec(m, ev["animal"], ev["tier"], ev["query"])
            r["uses"]+=1; r["scores"].append(float(ev["eff"]))
            r["q"]=round(r["q"]+(float(ev["eff"])-r["q"])/r["uses"],3)
            if r["uses"]>=MIN_TRIALS and _avg(r)<EFF_THRESH:
                r["blacklisted"]=True
            m["visits"][ev["animal"]]=m["visits"].get(ev["animal"],0)+1
            m["events"].append(ev)
        log("RL",f"{len(events)} episodes | {len(m['sounds'])} queries")
    except Exception as e:
        log("RL",f"Load error: {e}")
    return m

def save_rl(m):
    try:
        json.dump({"sounds":m["sounds"],"events":m["events"][-500:],
                   "visits":m["visits"]},
                  open(RL_FILE,"w",encoding="utf-8"),indent=2)
    except Exception as e:
        log("RL",f"Save error: {e}")

_last_q = {}

def pick_query(m, animal, tier):
    all_q = SOUNDS[animal][tier]["q"]
    avail = [q for q in all_q
             if not get_rec(m,animal,tier,q)["blacklisted"]
             and q!=_last_q.get(animal)]
    if not avail:
        avail=[q for q in all_q if not get_rec(m,animal,tier,q)["blacklisted"]]
    if not avail:
        for q in all_q: get_rec(m,animal,tier,q)["blacklisted"]=False
        avail=list(all_q)
    unseen=[q for q in avail if get_rec(m,animal,tier,q)["uses"]==0]
    if unseen:
        chosen=unseen[0]; log("RL",f"FIRST  t{tier} → '{chosen}'")
    else:
        total=max(2,sum(get_rec(m,animal,tier,q)["uses"] for q in all_q))
        chosen=max(avail,key=lambda q:
            get_rec(m,animal,tier,q)["q"]
            +UCB_C*math.sqrt(2.0*math.log(total)
                             /max(1,get_rec(m,animal,tier,q)["uses"])))
        r=get_rec(m,animal,tier,chosen)
        log("RL",f"UCB1   t{tier} → '{chosen}' q={r['q']:.3f} n={r['uses']}")
    _last_q[animal]=chosen
    return chosen

def update_rl(m, animal, query, tier, absent_s, played_s):
    eff=round(min(1.0, absent_s/max(0.1,played_s)),3)
    r=get_rec(m,animal,tier,query)
    r["uses"]+=1; r["scores"].append(eff)
    r["q"]=round(r["q"]+(eff-r["q"])/r["uses"],3)
    avg=_avg(r)
    if r["uses"]>=MIN_TRIALS and avg<EFF_THRESH:
        r["blacklisted"]=True; log("RL",f"BLACKLIST '{query}' avg={avg:.3f}")
    m["visits"][animal]=m["visits"].get(animal,0)+1
    ev={"ts":time.strftime("%Y-%m-%d %H:%M:%S"),
        "animal":animal,"query":query,"tier":tier,
        "eff":eff,"avg":avg,"q":r["q"],"uses":r["uses"],
        "played_s":round(played_s,2),"absent_s":round(absent_s,2)}
    m["events"].append(ev); save_rl(m)
    log("RESULT",
        f"eff={eff:.3f} ({'EXCELLENT' if eff>.7 else 'GOOD' if eff>.4 else 'POOR'}) "
        f"avg={avg:.3f} uses={r['uses']} absent={absent_s:.1f}s played={played_s:.1f}s")

# ── SOUND CACHE ───────────────────────────────────────────────────────────────
_cache, _clk    = {}, threading.Lock()
_net_ok         = True
_tmp            = []
_ready          = {a: threading.Event() for a in ANIMALS}
_inflight, _ilk = set(), threading.Lock()

def _ck(a,t,q):    return f"{a}::{t}::{q}"
def _fname(a,t,q): return f"{a}__{t}__{q.replace(' ','_')[:60]}.wav"

def _cached_entries(animal, tiers=None):
    tiers_list=[tiers] if isinstance(tiers,int) else (tiers or [1,2,3])
    rows=[]
    with _clk:
        for key,path in list(_cache.items()):
            parts=key.split("::",2)
            if len(parts)!=3: continue
            a,tier_s,query=parts
            try: tier=int(tier_s)
            except ValueError: continue
            if a==animal and tier in tiers_list and os.path.exists(path):
                rows.append((tier,query,path))
    return rows

def has_cached_sound(animal, tier=None):
    return bool(_cached_entries(animal,tier))

def _load_disk():
    if not os.path.exists(CACHE_DIR): return 0
    n=0
    for name in os.listdir(CACHE_DIR):
        if not name.lower().endswith(".wav"): continue
        stem=os.path.splitext(name)[0]
        parts=stem.split("__",2)
        if len(parts)!=3: continue
        animal,tier_s,query_s=parts
        if animal not in ANIMALS: continue
        try: tier=int(tier_s)
        except ValueError: continue
        p=os.path.join(CACHE_DIR,name)
        if os.path.getsize(p)<=1000: continue
        with _clk: _cache[_ck(animal,tier,query_s.replace("_"," "))]=p
        n+=1
    return n

def _save_disk(a,t,q,wav):
    try:
        os.makedirs(CACHE_DIR,exist_ok=True)
        dst=os.path.join(CACHE_DIR,_fname(a,t,q))
        if not os.path.exists(dst): shutil.copy2(wav,dst)
    except Exception: pass

def _name_ok(name, ban):
    n=name.lower()
    return not any(b in n for b in ban)

def _mp3_to_audio(mp3_bytes):
    mp3=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3")
    mp3.write(mp3_bytes); mp3.close(); _tmp.append(mp3.name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pydub import AudioSegment
            a=(AudioSegment.from_mp3(mp3.name)
               .set_frame_rate(44100).set_channels(2).set_sample_width(2))
        wf=tempfile.NamedTemporaryFile(delete=False,suffix=".wav"); wf.close()
        a.export(wf.name,format="wav"); _tmp.append(wf.name)
        return wf.name
    except Exception: pass
    try:
        _init_mixer()
        snd=pygame.mixer.Sound(mp3.name)
        arr=pygame.sndarray.array(snd)
        if arr.ndim==1: arr=np.column_stack([arr,arr])
        arr=arr.astype(np.int16)
        wf=tempfile.NamedTemporaryFile(delete=False,suffix=".wav"); wf.close()
        with wave.open(wf.name,"w") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100)
            w.writeframes(arr.tobytes())
        _tmp.append(wf.name); return wf.name
    except Exception: pass
    return mp3.name

def _download(animal, tier, query):
    global _net_ok
    k=_ck(animal,tier,query)
    with _clk:
        if k in _cache: return _cache[k]
    ban=SOUNDS[animal][tier]["ban"]
    try:
        r=requests.get("https://freesound.org/apiv2/search/text/",
            params={"query":query,"token":FS_KEY,
                    "fields":"id,name,duration,previews",
                    "filter":"duration:[5 TO 45]",
                    "sort":"rating_desc","page_size":20},
            timeout=12)
        r.raise_for_status()
        for res in r.json().get("results",[]):
            if not _name_ok(res.get("name",""),ban): continue
            url=(res["previews"].get("preview-hq-mp3")
                 or res["previews"].get("preview-lq-mp3"))
            if not url: continue
            r2=requests.get(url,timeout=20); r2.raise_for_status()
            wav=_mp3_to_audio(r2.content)
            if wav:
                with _clk: _cache[k]=wav
                _net_ok=True
                log("SOUND",
                    f"'{res['name'][:50]}' → '{query}'  {res.get('duration',0):.1f}s")
                _save_disk(animal,tier,query,wav); return wav
        log("SEARCH",f"No result for '{query}'")
    except Exception as e:
        _net_ok=False; log("SEARCH",f"{type(e).__name__} '{query}'")
    return None

def _preload_animal(animal,m):
    tier=get_tier(m["visits"].get(animal,0))
    if has_cached_sound(animal,tier): _ready[animal].set(); return
    if has_cached_sound(animal):      _ready[animal].set()
    with _ilk:
        if animal in _inflight: return
        _inflight.add(animal)
    def _work():
        ok=[False]; lk2=threading.Lock()
        def _one(q):
            if _download(animal,tier,q):
                with lk2: ok[0]=True; _ready[animal].set()
        ts=[threading.Thread(target=_one,args=(q,),daemon=True)
            for q in SOUNDS[animal][tier]["q"]]
        for t in ts: t.start()
        for t in ts: t.join()
        if not ok[0]:
            if has_cached_sound(animal): _ready[animal].set()
            else: log("AGENT",f"All downloads failed: {animal} t{tier}")
        with _ilk: _inflight.discard(animal)
    threading.Thread(target=_work,daemon=True).start()

def start_download(animal,m): _preload_animal(animal,m)

def get_wav(animal,tier,preferred,m):
    qs=list(SOUNDS[animal][tier]["q"])
    if preferred in qs: qs.remove(preferred); qs.insert(0,preferred)
    for q in qs:
        with _clk: path=_cache.get(_ck(animal,tier,q))
        if path and os.path.exists(path): return path,q,tier
    fallbacks=[]
    for at,aq,ap in _cached_entries(animal):
        if at==tier: continue
        rec=get_rec(m,animal,at,aq)
        fallbacks.append(((rec["uses"]>0,rec["q"],-abs(at-tier)),at,aq,ap))
    if fallbacks:
        _,at,aq,ap=max(fallbacks,key=lambda x:x[0])
        return ap,aq,at
    return None,None,None

# ── AUDIO ─────────────────────────────────────────────────────────────────────
_mxlk   = threading.Lock()
_sd_dev = None

def _init_mixer():
    with _mxlk:
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100,-16,2,2048); pygame.mixer.init()

def _find_sd():
    try:
        import sounddevice as sd
        idx=sd.default.device[1]
        if idx>=0 and sd.query_devices(idx)["max_output_channels"]>0: return idx
        for i,d in enumerate(sd.query_devices()):
            if d["max_output_channels"]>0: return i
    except Exception: pass
    return None

def play_loop(path,stop_evt):
    if not path or not os.path.exists(path):
        log("AUDIO","File missing"); return
    t0=time.time(); ext=os.path.splitext(path)[1].lower()
    if _sd_dev is not None and ext==".wav":
        try:
            import sounddevice as sd
            with wave.open(path) as wf:
                sr,ch=wf.getframerate(),wf.getnchannels()
                arr=(np.frombuffer(wf.readframes(wf.getnframes()),np.int16)
                     .reshape(-1,ch).astype(np.float32)/32768.0)
            if ch==1: arr=np.column_stack([arr,arr])
            arr=np.clip(arr*3.5,-1.0,1.0); clip=len(arr)/sr
            log("AUDIO",f"sounddevice +11dB  {clip:.1f}s  looping")
            while not stop_evt.is_set() and time.time()-t0<MAX_PLAY:
                sd.play(arr,sr,device=_sd_dev,blocking=False)
                end=time.time()+clip
                while time.time()<end:
                    if stop_evt.is_set() or time.time()-t0>=MAX_PLAY:
                        sd.stop(); return
                    time.sleep(0.04)
            sd.stop(); return
        except Exception as e: log("AUDIO",f"sounddevice: {e}")
    try:
        _init_mixer()
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play(loops=-1)
        log("AUDIO",f"pygame loops=-1  max={MAX_PLAY}s")
        while not stop_evt.is_set() and time.time()-t0<MAX_PLAY:
            time.sleep(0.05)
        pygame.mixer.music.stop()
    except Exception as e: log("AUDIO",f"pygame: {e}")

# ── VISION ────────────────────────────────────────────────────────────────────
def load_models():
    yolo=YOLOModel(YOLO_MODEL)
    yolo.to("cpu")
    log("STARTUP",f"YOLO: {YOLO_MODEL} — detect only, CPU, MKL-DNN off")
    net=models.resnet50()
    net.fc=nn.Linear(net.fc.in_features,len(CLASSES))
    net.load_state_dict(torch.load(RESNET_PATH,map_location=DEVICE))
    net=net.to(DEVICE).eval()
    log("STARTUP",
        f"ResNet50 {len(CLASSES)} classes {DEVICE} — "
        f"conf>={detector.CONF_THRESH} ent<={detector.ENT_THRESH} gap>={detector.GAP_THRESH}")
    return yolo,net

_tracker = detector.MotionTracker()
_mog2    = detector.MOG2Detector()
_det_t   = [0.0]

def classify(model_tuple, frame):
    yolo,net=model_tuple
    dets=detector.process_frame(
        yolo,net,_tracker,frame,
        mog2=_mog2,draw=True,allow_scan=False)

    if not dets:
        return "no animal",0.0,0.0

    best=max(dets,key=lambda d:d["conf"])
    if time.time()-_det_t[0]>2.0:
        _det_t[0]=time.time()
        log("DETECT",
            f"{best['name']} {best['conf']:.1%}  ent={best['ent']:.2f}  "
            f"src={best['source']}  moving={best['moving']}")
    return best["name"],best["conf"],best["ent"]

# ── SHARED STATE ──────────────────────────────────────────────────────────────
shared     = {"animal":"no animal","lock":threading.Lock()}
first_seen = {}; last_seen = {}

def track(animal,now):
    if animal!="no animal":
        first_seen.setdefault(animal,now); last_seen[animal]=now
    else:
        for a in [a for a,ts in list(last_seen.items()) if now-ts>3]:
            first_seen.pop(a,None); last_seen.pop(a,None)

def stay_pre(animal,t0):
    return max(0.0,t0-first_seen[animal]) if animal in first_seen else 0.0

last_triggered={}; cur_query=defaultdict(str); cur_tier=defaultdict(int)

def can_trigger(a):
    now=time.time()
    if now-last_triggered.get(a,0)>COOLDOWN:
        last_triggered[a]=now; return True
    return False

def _explain(animal,query):
    try:
        r=groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":
                f"One sentence: why does '{query}' biologically repel {animal}? "
                f"({SCIENCE[animal]})"}],
            max_tokens=80,temperature=0.3)
        log("LLM",r.choices[0].message.content.strip())
    except Exception as e: log("LLM",str(e))

# ── RL AGENT ──────────────────────────────────────────────────────────────────
def run_agent(animal,m):
    visits=m["visits"].get(animal,0); tier=get_tier(visits)
    section(f"DETECTION: {animal}  tier={tier}  visit#{visits}")
    log("SCIENCE",SCIENCE[animal])

    if has_cached_sound(animal): _ready[animal].set()
    if not _ready[animal].wait(timeout=10):
        log("AGENT","Sound not ready — skipping"); return

    chosen=pick_query(m,animal,tier)
    wav,query,play_tier=get_wav(animal,tier,chosen,m)
    if not wav: log("AGENT","No cached sound"); return
    if play_tier!=tier: log("RL",f"Fallback t{play_tier} '{query}'")
    elif query!=chosen: log("RL",f"Using '{query}' (preferred '{chosen}')")

    cur_query[animal]=query; cur_tier[animal]=play_tier
    sound_start=time.time()
    log("SOUND",f"'{query}' | MIN={MIN_PLAY_S}s  MAX={MAX_PLAY}s")
    threading.Thread(target=_explain,args=(animal,query),daemon=True).start()

    stop=threading.Event()
    threading.Thread(target=play_loop,args=(wav,stop),daemon=True).start()

    absent_s=0.0; absent_since=None; last_tick=time.time()

    while time.time()-sound_start<MAX_PLAY:
        time.sleep(0.15)
        now=time.time(); dt=now-last_tick; last_tick=now
        elapsed=now-sound_start
        with shared["lock"]: here=(shared["animal"]==animal)

        if here:
            absent_since=None
        else:
            absent_s+=dt
            if absent_since is None:
                absent_since=now
                log("MONITOR",f"Animal left at {elapsed:.1f}s")
            elif elapsed>=MIN_PLAY_S and now-absent_since>=STOP_ABSENT:
                log("MONITOR",f"MIN {MIN_PLAY_S}s + absent {STOP_ABSENT}s → stop")
                break

    stop.set()
    played_s=max(0.1,time.time()-sound_start)
    update_rl(m,animal,query,play_tier,absent_s,played_s)
    first_seen.pop(animal,None); last_seen.pop(animal,None)

# ── OVERLAY ───────────────────────────────────────────────────────────────────
def draw_overlay(frame,m,animal,sb,busy,query,query_tier):
    h,w=frame.shape[:2]
    cv2.rectangle(frame,(0,h-72),(w,h),(15,15,15),-1)
    if animal=="no animal":
        cv2.putText(frame,
                    f"No animal | {'NET:OK' if _net_ok else 'OFFLINE'}",
                    (10,h-28),cv2.FONT_HERSHEY_SIMPLEX,0.48,(140,140,140),1)
    else:
        visits=m["visits"].get(animal,0)
        tier=query_tier or get_tier(visits)
        tcol={1:(80,255,80),2:(0,200,255),3:(60,80,255)}[tier]
        r=get_rec(m,animal,tier,query) if query else {}
        e=_avg(r) if r else 0.5
        ecol=(80,255,80) if e>0.6 else (0,200,255) if e>0.35 else (60,80,255)
        cv2.putText(frame,
                    f"Tier {tier} | visit#{visits} | eff={e:.2f} uses={r.get('uses',0)}",
                    (10,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.38,tcol,1)
        cv2.putText(frame,
                    f"'{query}' stay={sb:.1f}s | {'NET:OK' if _net_ok else 'OFFLINE'}",
                    (10,h-30),cv2.FONT_HERSHEY_SIMPLEX,0.38,ecol,1)
    if busy:
        cv2.putText(frame,">>> REPELLING — SOUND ACTIVE <<<",
                    (10,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.48,(50,255,120),2)
    return frame

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    section(f"AI Scarecrow v7  |  {DEVICE}")
    m=load_rl()

    n=_load_disk()
    if n:
        log("STARTUP",f"{n} sounds from disk cache")
        for a in ANIMALS:
            if has_cached_sound(a): _ready[a].set()

    log("STARTUP","Preloading sounds for all animals in background...")
    for a in ANIMALS:
        if not has_cached_sound(a): _preload_animal(a,m)

    _init_mixer()
    global _sd_dev; _sd_dev=_find_sd()
    log("STARTUP",
        f"Audio: {'sounddevice dev='+str(_sd_dev) if _sd_dev is not None else 'pygame'}")

    try:
        model_tuple=load_models(); log("STARTUP","Models ready")
    except FileNotFoundError:
        model_tuple=None; log("STARTUP","Model file not found — DEMO mode")

    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): cap=cv2.VideoCapture(1)

    consec={}; prev="no animal"; agent_thread=None; frame_n=0
    hold_animal="no animal"; hold_until=0.0; active="no animal"
    last_raw,last_conf,last_ent="no animal",0.0,0.0

    TEST={ord(str(i+1)):a for i,a in enumerate(ANIMALS)}
    log("STARTUP","Keys: 1-8=test animal | s=stats | r=reset blacklists | q=quit")

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame_n+=1; now=time.time()
        busy=agent_thread is not None and agent_thread.is_alive()

        if frame_n%CLASSIFY_N==0:
            if model_tuple:
                last_raw,last_conf,last_ent=classify(model_tuple,frame)
            else:
                last_raw=random.choice(CLASSES)
                last_conf=random.uniform(0.85,0.99); last_ent=0.4

        consec={k:v+1 if k==last_raw else 0 for k,v in consec.items()}
        consec={k:v for k,v in consec.items() if v>0}
        consec.setdefault(last_raw,1)

        raw_confirmed=next(
            (a for a,n in consec.items() if n>=TRIG_FRAMES and a!="no animal"),
            "no animal")

        if raw_confirmed!="no animal":
            hold_animal=raw_confirmed; hold_until=now+CONFIRM_HOLD_S

        confirmed=(hold_animal
                   if hold_animal!="no animal" and now<=hold_until
                   else "no animal")

        with shared["lock"]: shared["animal"]=confirmed
        track(confirmed,now)
        if prev!="no animal" and confirmed=="no animal":
            first_seen.pop(prev,None); last_seen.pop(prev,None)
        prev=confirmed

        if last_raw!="no animal": start_download(last_raw,m)

        if not busy: active="no animal"
        disp=active if busy else confirmed
        sb=stay_pre(disp,now) if disp!="no animal" else 0.0

        # HUD
        cv2.rectangle(frame,(8,8),(frame.shape[1]-8,92),(0,0,0),-1)
        cv2.putText(frame,
                    f"ResNet: {last_raw} {last_conf:.0%} ent={last_ent:.2f}",
                    (14,30),cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,100,100),1)
        col=(0,60,255) if disp!="no animal" else (0,200,0)
        cv2.putText(frame,
                    f"CONFIRMED: {disp if disp!='no animal' else 'none'}"
                    f"  [{consec.get(last_raw,0)}/{TRIG_FRAMES}]",
                    (14,70),cv2.FONT_HERSHEY_SIMPLEX,0.78,col,2)

        if disp!="no animal":
            cv2.rectangle(frame,(8,92),(frame.shape[1]-8,118),
                          (0,110,0) if busy else (100,0,0),-1)
            cv2.putText(frame,
                        "REPELLING..." if busy else f"INTRUDER stay={sb:.0f}s",
                        (14,112),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),2)

        draw_overlay(frame,m,disp,sb,busy,cur_query[disp],cur_tier[disp])
        cv2.imshow("AI Scarecrow [RL+YOLO+MOG2] v7",frame)

        if confirmed!="no animal" and not busy and can_trigger(confirmed):
            active=confirmed
            agent_thread=threading.Thread(
                target=run_agent,args=(confirmed,m),daemon=True)
            agent_thread.start()

        key=cv2.waitKey(1)&0xFF
        if key==ord("q"): break
        elif key==ord("r"):
            for r2 in m["sounds"].values(): r2["blacklisted"]=False
            save_rl(m); log("RL","Blacklists cleared")
        elif key==ord("s"):
            with _plk:
                print(f"\n{'='*60}  RL STATS")
                for a in ANIMALS:
                    evs=[e for e in m["events"] if e["animal"]==a]
                    if evs:
                        print(f"\n  {a}  tier={get_tier(m['visits'].get(a,0))}"
                              f"  visits={m['visits'].get(a,0)}")
                        for e in evs[-3:]:
                            print(f"    t{e.get('tier')} '{e.get('query')}'"
                                  f"  eff={e.get('eff',0):.3f}"
                                  f"  played={e.get('played_s',0):.1f}s")
                print("="*60)
        elif key in TEST:
            ta=TEST[key]; last_triggered[ta]=0
            first_seen[ta]=now-10; last_seen[ta]=now
            consec[ta]=TRIG_FRAMES; hold_animal=ta; hold_until=now+CONFIRM_HOLD_S
            with shared["lock"]: shared["animal"]=ta
            start_download(ta,m)
            if not busy:
                active=ta
                agent_thread=threading.Thread(
                    target=run_agent,args=(ta,m),daemon=True)
                agent_thread.start()

    cap.release(); cv2.destroyAllWindows()
    try: pygame.mixer.music.stop(); pygame.mixer.quit()
    except Exception: pass
    save_rl(m)
    for f in _tmp:
        try: os.remove(f)
        except Exception: pass
    log("DONE",f"Saved → {RL_FILE}")


if __name__=="__main__":
    main()