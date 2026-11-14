from pathlib import Path
import sys,json,cv2,numpy as np,shutil
from ultralytics import YOLO
base=Path(__file__).resolve().parent.parent
l3=base/'3labO'; l4=base/'4labO'
model=YOLO(str(l3/'runs/yolo11m_rps/weights/best.pt'))
images=l3/'dataset/test/images'; labels=l3/'dataset/test/labels'
chosen={}; crops={}; records=[]; detections=[]
for label in sorted(labels.glob('*.txt')):
 lines=[list(map(float,s.split())) for s in label.read_text().splitlines() if s.strip()]
 if len(lines)!=1 or int(lines[0][0]) in chosen: continue
 c,xc,yc,bw,bh=lines[0]; c=int(c)
 if bw<.18 or bh<.18: continue
 p=images/(label.stem+'.jpg')
 if not p.exists(): continue
 im=cv2.imread(str(p)); h,w=im.shape[:2]
 r=model.predict(im,conf=.5,device=0,verbose=False)[0]
 matches=[b for b in r.boxes if int(b.cls)==c and float(b.conf)>.85]
 if not matches: continue
 b=max(matches,key=lambda b:float(b.conf)); chosen[c]=str(p)
 x1=max(0,int((xc-bw*.7)*w));x2=min(w,int((xc+bw*.7)*w))
 y1=max(0,int((yc-bh*.7)*h));y2=min(h,int((yc+bh*.7)*h))
 crop=im[y1:y2,x1:x2]; crops[c]=cv2.resize(crop,(180,180))
 out=l3/'results/detections'; out.mkdir(exist_ok=True)
 dst=out/f'example_{c+1}_det.jpg'; cv2.imencode('.jpg',r.plot(line_width=2))[1].tofile(str(dst))
 records.append({'class':r.names[c],'source':str(p),'image':str(dst),'confidence':float(b.conf)})
 detections.append(p.name+': найдено объектов '+str(len(r.boxes)))
 for bb in r.boxes:
  detections.append(f'  {r.names[int(bb.cls)]}: уверенность {float(bb.conf):.3f}; рамка '+str([round(v) for v in bb.xyxy[0].tolist()]))
 if len(chosen)==3: break
assert len(chosen)==3,chosen
out=l3/'results'; (out/'detection_examples.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'detection_console.txt').write_text('\n'.join(detections),encoding='utf-8')
(l4/'weights').mkdir(exist_ok=True); shutil.copy2(l3/'runs/yolo11m_rps/weights/best.pt',l4/'weights/best.pt')
(l4/'videos').mkdir(exist_ok=True)
writer=cv2.VideoWriter(str(l4/'videos/gestures_control.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(640,480))
rng=np.random.default_rng(42); background=np.clip(rng.normal(100,6,(480,640,3)),0,255).astype(np.uint8)
for t in range(400):
 frame=background.copy()
 for c in range(3):
  if (c==0 and 100<=t<115) or (c==1 and 220<=t<265): continue
  x=int(225+190*np.sin(t/62+c*2.09)); y=int(145+110*np.cos(t/83+c*2.09))
  sz=int(170+15*np.sin(t/37+c)); patch=cv2.resize(crops[c],(sz,sz))
  frame[y:y+sz,x:x+sz]=patch
 cv2.putText(frame,'CONTROL VIDEO - test image crops',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
 writer.write(frame)
writer.release()
(l4/'videos/gestures_control_sources.json').write_text(json.dumps({'type':'synthetic motion of real held-out image crops','frames':400,'fps':25,'size':[640,480],'seed':42,'sources':chosen,'occlusions':{'0':[100,115],'1':[220,265]}},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2));print('\n'.join(detections))
