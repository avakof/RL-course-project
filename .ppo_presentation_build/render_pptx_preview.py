from pathlib import Path
import subprocess,time
from bs4 import BeautifulSoup
import pymupdf
from PIL import Image,ImageDraw
B=Path('/Users/alivakof/Desktop/RL/.ppo_presentation_build');Q=B/'quicklook'
subprocess.run(['qlmanage','-p','-o',str(Q),str(B/'PPO_Presentation_draft.pptx')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30)
D=Q/'PPO_Presentation_draft.pptx.qlpreview';s=BeautifulSoup((D/'Preview.html').read_text(),'html.parser')
for tag in s.find_all('img'):
 src=tag.get('src','')
 if src.endswith('.pdf'):
  pdf=pymupdf.open(D/src);pdf[0].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(D/(src+'.png'));tag['src']=src+'.png'
style=s.new_tag('style');style.string='@media print{@page{size:10in 5.625in;margin:0}html,body{margin:0!important;padding:0!important;width:960px!important;background:white!important}.slide{margin:0!important;width:960px!important;height:540px!important;page-break-after:always;break-after:page;overflow:hidden}.slide:last-child{page-break-after:auto}}'
s.head.append(style);preview=D/'native_preview.html';preview.write_text(str(s))
output=B/'native_preview.pdf';old=output.stat().st_mtime if output.exists() else 0
cmd=['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','--headless','--disable-gpu','--disable-background-networking','--disable-component-update','--no-pdf-header-footer','--user-data-dir='+str(B/'chrome-pptx-profile'),'--print-to-pdf='+str(output),preview.as_uri()]
with (B/'native_chrome.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True)
 try:
  for _ in range(80):
   time.sleep(.25)
   if output.exists() and output.stat().st_mtime>old and 'bytes written' in (B/'native_chrome.log').read_text():break
  else:raise RuntimeError('Native preview print timeout')
 finally:
  proc.terminate()
  try:proc.wait(timeout=4)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
doc=pymupdf.open(output);print('Preview pages',len(doc));dest=B/'pptx_previews';dest.mkdir(exist_ok=True)
for i,p in enumerate(doc):p.get_pixmap(matrix=pymupdf.Matrix(.9,.9),alpha=False).save(dest/f'slide_{i+1:02}.png')
for batch in range(2):
 canvas=Image.new('RGB',(1440,1120),'#d9e1e7');d=ImageDraw.Draw(canvas)
 for k in range(12):
  i=batch*12+k+1;im=Image.open(dest/f'slide_{i:02}.png');im.thumbnail((460,259));x=(k%3)*480+10;y=(k//3)*280+15;canvas.paste(im,(x,y));d.text((x+5,y-12),str(i),fill='black')
 canvas.save(dest/f'montage_{batch+1}.png')
