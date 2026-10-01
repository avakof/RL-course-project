from pathlib import Path
import subprocess,time,os,json,hashlib,re,base64
from bs4 import BeautifulSoup
import pymupdf
from PIL import Image
R=Path('/Users/alivakof/Desktop/RL/PPO_Presentation');B=R.parent/'.ppo_presentation_build'
output=R/'PPO_Presentation.pdf';old=output.stat().st_mtime
cmd=['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','--headless','--disable-gpu','--disable-background-networking','--disable-component-update','--no-pdf-header-footer','--user-data-dir='+str(B/'chrome-profile'),'--print-to-pdf='+str(output),(R/'PPO_Presentation.html').as_uri()]
with (B/'chrome-final.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True)
 try:
  for _ in range(60):
   time.sleep(.25)
   if output.stat().st_mtime>old and 'bytes written to file' in (B/'chrome-final.log').read_text():break
  else:raise RuntimeError('PDF was not refreshed')
 finally:
  proc.terminate()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
D=pymupdf.open(output);assert len(D)==24
slides=json.loads((R/'data/slides.json').read_text())
issues=[]
for i,p in enumerate(D):
 txt=p.get_text();assert f'{i+1} / 24' in txt
 assert slides[i]['title'] in txt.replace('\n',' '),(i,txt[:200])
 for block in p.get_text('dict')['blocks']:
  if block['type']!=0:continue
  for line in block['lines']:
   for s in line['spans']:
    rect=pymupdf.Rect(s['bbox'])
    if not p.rect.contains(rect):issues.append((i+1,s['text'],list(rect)))
 p.get_pixmap(matrix=pymupdf.Matrix(.8,.8),alpha=False).save(B/'previews'/f'slide_{i+1:02}.png')
assert not issues,issues
for name in ['PPO_Presentation.html','PPO_Analysis.html']:
 soup=BeautifulSoup((R/name).read_text(),'html.parser')
 imgs=soup.find_all('img');assert len(imgs)==11
 for im in imgs:
  assert im['src'].startswith('data:image/png;base64,')
  assert base64.b64decode(im['src'].split(',')[1]).startswith(b'\x89PNG')
 assert not soup.find_all('script',src=True)
 assert not soup.find_all('link',rel='stylesheet')
source=R.parent/'Project_PPO_All_In_One_Fixed.ipynb'
audit=json.loads((R/'data/audit.json').read_text());assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['source_sha256'],'Source notebook changed during analysis'
validation={'pdf_pages':len(D),'chart_count':11,'standalone_html_images_verified':True,'pdf_text_within_page':True,'source_notebook_unchanged':True,'material_scope':'Captured experiment analysis; frozen-policy replays; no new training.'}
(B/'delivery_validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps(validation,indent=2))
