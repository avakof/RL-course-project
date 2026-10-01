from pathlib import Path
from io import BytesIO
import json,zipfile,math,ast,copy,re,html
from xml.etree import ElementTree as E
import numpy as np,pandas as pd
import pymupdf
from bs4 import BeautifulSoup
R=Path('/Users/alivakof/Desktop/RL/PPO_Presentation');B=R.parent/'.ppo_presentation_build'
P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';C='http://schemas.openxmlformats.org/drawingml/2006/chart';REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships';PR='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types';X='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
for p,u in [('p',P),('a',A),('c',C),('r',REL)]:E.register_namespace(p,u)
E.register_namespace('',CT)
def q(ns,tag):return '{'+ns+'}'+tag
def node(parent,ns,tag,attrs=None,txt=None):
 n=E.SubElement(parent,q(ns,tag),{k:str(v) for k,v in (attrs or {}).items()});n.text=txt;return n
def xml(e):return E.tostring(e,encoding='utf-8',xml_declaration=True)
def em(v):return round(float(v)*12700)
def solid(n,color):node(node(n,A,'solidFill'),A,'srgbClr',{'val':color})
def ln(n,color='D5DFE6',width=.6):
 l=node(n,A,'ln',{'w':em(width)});solid(l,color);return l
def rels(items):
 r=E.Element(q(PR,'Relationships'))
 for rid,typ,target in items:node(r,PR,'Relationship',{'Id':rid,'Type':REL+'/'+typ,'Target':target})
 return xml(r)
def group(root):
 sp=node(root,P,'spTree');nv=node(sp,P,'nvGrpSpPr');node(nv,P,'cNvPr',{'id':1,'name':''});node(nv,P,'cNvGrpSpPr');node(nv,P,'nvPr');xf=node(node(sp,P,'grpSpPr'),A,'xfrm')
 for tag,atts in [('off',{'x':0,'y':0}),('ext',{'cx':0,'cy':0}),('chOff',{'x':0,'y':0}),('chExt',{'cx':0,'cy':0})]:node(xf,A,tag,atts)
 return sp
def txbody(parent,tag='txBody',ns=P):
 tb=node(parent,ns,tag);node(tb,A,'bodyPr',{'wrap':'none','lIns':0,'rIns':0,'tIns':0,'bIns':0,'anchor':'t'});node(tb,A,'lstStyle');return tb
def para(tb,text,size=18,color='142735',bold=False,font='Arial'):
 p=node(tb,A,'p');node(p,A,'pPr',{'marL':0,'marR':0,'indent':0});run=node(p,A,'r');rp=node(run,A,'rPr',{'lang':'en-US','sz':round(size*100),'b':int(bold),'dirty':0});solid(rp,color);node(rp,A,'latin',{'typeface':font});node(run,A,'t',txt=text);node(p,A,'endParaRPr',{'lang':'en-US','sz':round(size*100)});return p
class Slide:
 def __init__(self,num,bg='FFFFFF'):
  self.num=num;self.root=E.Element(q(P,'sld'),{'showMasterSp':'0'});cs=node(self.root,P,'cSld');bp=node(node(cs,P,'bg'),P,'bgPr');solid(bp,bg);node(bp,A,'effectLst');self.sp=group(cs);ov=node(self.root,P,'clrMapOvr');node(ov,A,'masterClrMapping');self.id=1;self.rels=[('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','notesSlide',f'../notesSlides/notesSlide{num}.xml')]
 def shape(self,x,y,w,h,name,geom='rect'):
  self.id+=1;sp=node(self.sp,P,'sp');nv=node(sp,P,'nvSpPr');node(nv,P,'cNvPr',{'id':self.id,'name':name});node(nv,P,'cNvSpPr',{'txBox':int(geom=='textbox')});node(nv,P,'nvPr');pr=node(sp,P,'spPr');xf=node(pr,A,'xfrm');node(xf,A,'off',{'x':em(x),'y':em(y)});node(xf,A,'ext',{'cx':em(w),'cy':em(h)});geo=node(pr,A,'prstGeom',{'prst':'rect'});node(geo,A,'avLst');return sp,pr
 def rect(self,r,color):
  sp,pr=self.shape(r.x0,r.y0,r.width,r.height,'Rule');solid(pr,color);node(node(pr,A,'ln'),A,'noFill')
 def textline(self,line):
  x,y,x1,y1=line['bbox'];h=y1-y;w=x1-x
  w = max(w+5, 915-x) if max(s['size'] for s in line['spans']) >= 25 else w+5
  sp,pr=self.shape(x,y+.2,w,h+3,'Text','textbox');node(pr,A,'noFill');node(node(pr,A,'ln'),A,'noFill');tb=txbody(sp);p=node(tb,A,'p');node(p,A,'pPr',{'marL':0,'indent':0})
  for span in line['spans']:
   if not span['text']:continue
   r=node(p,A,'r');atts={'lang':'en-US','sz':round(span['size']*100),'b':int(bool(span['flags']&16)),'i':int(bool(span['flags']&2)),'dirty':0}
   rp=node(r,A,'rPr',atts);solid(rp,f'{span["color"]:06X}');font='Times New Roman' if 'Times' in span['font'] else 'Arial';node(rp,A,'latin',{'typeface':font});node(r,A,'t',txt=span['text'])
  node(p,A,'endParaRPr',{'lang':'en-US','sz':round(line['spans'][-1]['size']*100)})
 def graphic(self,x,y,w,h,name,uri):
  self.id+=1;gf=node(self.sp,P,'graphicFrame');nv=node(gf,P,'nvGraphicFramePr');node(nv,P,'cNvPr',{'id':self.id,'name':name});node(nv,P,'cNvGraphicFramePr');node(nv,P,'nvPr');xf=node(gf,P,'xfrm');node(xf,A,'off',{'x':em(x),'y':em(y)});node(xf,A,'ext',{'cx':em(w),'cy':em(h)});return node(node(gf,A,'graphic'),A,'graphicData',{'uri':uri})
 def table(self,rows,cols,ys):
  x=cols[0];y=ys[0];g=self.graphic(x,y,cols[-1]-x,ys[-1]-y,'Editable table',A+'/table');t=node(g,A,'tbl');node(t,A,'tblPr',{'firstRow':0,'bandRow':0});grid=node(t,A,'tblGrid')
  for l,r in zip(cols,cols[1:]):node(grid,A,'gridCol',{'w':em(r-l)})
  for i,row in enumerate(rows):
   tr=node(t,A,'tr',{'h':em(ys[i+1]-ys[i])});ci=0
   for cell in row:
    span=int(cell.get('colspan',1));ishead=cell.name=='th';attrs={'gridSpan':span} if span>1 else {};tc=node(tr,A,'tc',attrs);tb=txbody(tc,ns=A);tb.find(q(A,'bodyPr')).set('wrap','square');para(tb,cell.get_text(' ',strip=True),17.25,'2166AC' if ishead else '142735',ishead)
    pr=node(tc,A,'tcPr',{'marL':em(9.75),'marR':em(9.75),'marT':em(11.25),'marB':em(10.5),'anchor':'t'})
    # Schema order: cell borders precede fill.
    line=node(pr,A,'lnB',{'w':em(.6)});solid(line,'D5DFE6')
    solid(pr,'EFF4F7' if i==0 and ishead else 'FFFFFF');ci+=1
    for _ in range(span-1):
     tc=node(tr,A,'tc',{'hMerge':1});tb=txbody(tc,ns=A);para(tb,'',17.25);pr=node(tc,A,'tcPr');lnb=node(pr,A,'lnB',{'w':em(.6)});solid(lnb,'D5DFE6');solid(pr,'FFFFFF');ci+=1
   assert ci==len(cols)-1
 def chart(self,spec,box):
  global chart_count
  chart_count+=1;k=chart_count;x,y,w,h=box;rid=f'rId{len(self.rels)+1}';self.rels.append((rid,'chart',f'../charts/chart{k}.xml'));gd=self.graphic(x,y,w,h,spec['title'],C);node(gd,C,'chart',{q(REL,'id'):rid});chartpart,book=make_chart(spec,k);parts[f'ppt/charts/chart{k}.xml']=chartpart;parts[f'ppt/charts/_rels/chart{k}.xml.rels']=rels([('rId1','package',f'../embeddings/chart{k}.xlsx')]);parts[f'ppt/embeddings/chart{k}.xlsx']=book;chart_specs.append({'slide':self.num,'chart':k,**spec})
def col(n):
 s=''
 while n:n,r=divmod(n-1,26);s=chr(65+r)+s
 return s
def workbook(cats,series):
 z=BytesIO();sheet=E.Element(q(X,'worksheet'));data=node(sheet,X,'sheetData');rows=[['Category']+[s['name'] for s in series]]+[[cat]+[s['values'][i] for s in series] for i,cat in enumerate(cats)]
 for ri,vals in enumerate(rows,1):
  row=node(data,X,'row',{'r':ri})
  for ci,v in enumerate(vals,1):
   if v is None:continue
   ce=node(row,X,'c',{'r':f'{col(ci)}{ri}'})
   if isinstance(v,(int,float,np.floating)):
    node(ce,X,'v',txt=format(float(v),'.16g'))
   else:
    ce.set('t','inlineStr');node(node(ce,X,'is'),X,'t',txt=str(v))
 wb=E.Element(q(X,'workbook'));sht=node(wb,X,'sheets');node(sht,X,'sheet',{'name':'Data','sheetId':1,q(REL,'id'):'rId1'})
 types=E.Element(q(CT,'Types'));node(types,CT,'Default',{'Extension':'rels','ContentType':'application/vnd.openxmlformats-package.relationships+xml'});node(types,CT,'Default',{'Extension':'xml','ContentType':'application/xml'})
 for path,typ in [('xl/workbook.xml','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'),('xl/worksheets/sheet1.xml','application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')]:node(types,CT,'Override',{'PartName':'/'+path,'ContentType':typ})
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
  for path,payload in {'[Content_Types].xml':xml(types),'_rels/.rels':rels([('rId1','officeDocument','xl/workbook.xml')]),'xl/workbook.xml':xml(wb),'xl/_rels/workbook.xml.rels':rels([('rId1','worksheet','worksheets/sheet1.xml')]),'xl/worksheets/sheet1.xml':xml(sheet)}.items():zz.writestr(path,payload)
 return z.getvalue()
def charttext(parent,text,size=14,bold=False):
 tb=node(parent,C,'txPr');node(tb,A,'bodyPr');node(tb,A,'lstStyle');para(tb,text,size,'142735',bold)
def title(parent,text,size=14):
 t=node(parent,C,'title');rich=node(node(t,C,'tx'),C,'rich');node(rich,A,'bodyPr');node(rich,A,'lstStyle');para(rich,text,size,'142735',True);node(t,C,'overlay',{'val':0});return t
def numref(parent,tag,values,formula,fmt):
 dest=node(parent,C,tag);nr=node(dest,C,'numRef');node(nr,C,'f',txt=formula);cache=node(nr,C,'numCache');node(cache,C,'formatCode',txt=fmt);node(cache,C,'ptCount',{'val':len(values)})
 for i,v in enumerate(values):
  if v is not None:node(node(cache,C,'pt',{'idx':i}),C,'v',txt=format(float(v),'.16g'))
def make_chart(s,k):
 s=copy.deepcopy(s)
 if '%' in s.get('fmt',''):
  for series_ in s['series']:series_['values']=[None if v is None else v*100 for v in series_['values']]
  for key in ['min','max']:
   if key in s:s[key]*=100
  s['fmt']='0.00';s['axis_fmt']='0.0';s['label_fmt']='0.00';s['title']+=' (%)'
 if s.get('log'):
  s['log']=False;s['title']=s['title'].replace(' (logarithmic scale)',' (USDT)')
 values_=[v for series_ in s['series'] for v in series_['values'] if v is not None]
 low_,high_=min(values_),max(values_);delta_=max(high_-low_,abs(high_)*.1,1e-6)
 s.setdefault('min',min(0,low_-delta_*.1));s.setdefault('max',high_+delta_*.2)
 cats=s['categories'];series=s['series'];fmt=s.get('fmt','0.00');horizontal=s.get('horizontal',False);n=len(cats);r=E.Element(q(C,'chartSpace'));node(r,C,'date1904',{'val':0});node(r,C,'lang',{'val':'en-US'});node(r,C,'roundedCorners',{'val':0});ch=node(r,C,'chart');title(ch,s['title'],s.get('title_size',14));node(ch,C,'autoTitleDeleted',{'val':0});plot=node(ch,C,'plotArea');node(plot,C,'layout')
 groups={}
 for idx,ss in enumerate(series):
  kind=ss.get('kind',s.get('kind','bar'))
  if kind not in groups:
   g=node(plot,C,'lineChart' if kind=='line' else 'barChart');groups[kind]=g
   if kind=='bar':node(g,C,'barDir',{'val':'bar' if horizontal else 'col'});node(g,C,'grouping',{'val':s.get('grouping','clustered')})
   else:node(g,C,'grouping',{'val':'standard'})
   node(g,C,'varyColors',{'val':0})
  g=groups[kind];se=node(g,C,'ser');node(se,C,'idx',{'val':idx});node(se,C,'order',{'val':idx});tx=node(se,C,'tx');sr=node(tx,C,'strRef');node(sr,C,'f',txt=f'Data!${col(idx+2)}$1');cache=node(sr,C,'strCache');node(cache,C,'ptCount',{'val':1});node(node(cache,C,'pt',{'idx':0}),C,'v',txt=ss['name'])
  sp=node(se,C,'spPr')
  if kind=='bar':
   if ss.get('hidden'):node(sp,A,'noFill')
   else:solid(sp,ss.get('color','2166AC'))
   node(node(sp,A,'ln'),A,'noFill');node(se,C,'invertIfNegative',{'val':0})
   for di,c in enumerate(ss.get('colors',[])):
    dp=node(se,C,'dPt');node(dp,C,'idx',{'val':di});dppr=node(dp,C,'spPr');solid(dppr,c);node(node(dppr,A,'ln'),A,'noFill')
  else:
   l=node(sp,A,'ln',{'w':em(ss.get('width',1.3))})
   if ss.get('no_line'):node(l,A,'noFill')
   else:solid(l,ss.get('color','2166AC'))
   if ss.get('dash'):node(l,A,'prstDash',{'val':'dash'})
   mk=node(se,C,'marker');node(mk,C,'symbol',{'val':'circle' if ss.get('marker') else 'none'})
   if ss.get('marker'):
    node(mk,C,'size',{'val':4});msp=node(mk,C,'spPr');solid(msp,ss.get('color','142735'));node(node(msp,A,'ln'),A,'noFill')
  cat=node(se,C,'cat');ref=node(cat,C,'strRef');node(ref,C,'f',txt=f'Data!$A$2:$A${n+1}');cache=node(ref,C,'strCache');node(cache,C,'ptCount',{'val':n})
  for i,v in enumerate(cats):node(node(cache,C,'pt',{'idx':i}),C,'v',txt=str(v))
  numref(se,'val',ss['values'],f'Data!${col(idx+2)}$2:${col(idx+2)}${n+1}',fmt)
  if kind=='line':node(se,C,'smooth',{'val':0})
 for kind,g in groups.items():
  if s.get('labels') and kind=='bar':
   dl=node(g,C,'dLbls');node(dl,C,'numFmt',{'formatCode':s.get('label_fmt',fmt),'sourceLinked':0});charttext(dl,'',10);node(dl,C,'dLblPos',{'val':'outEnd'});node(dl,C,'showLegendKey',{'val':0});node(dl,C,'showVal',{'val':1});node(dl,C,'showCatName',{'val':0});node(dl,C,'showSerName',{'val':0})
  if kind=='bar':node(g,C,'gapWidth',{'val':s.get('gap',70)});node(g,C,'overlap',{'val':100 if s.get('grouping')=='stacked' else 0})
  node(g,C,'axId',{'val':100});node(g,C,'axId',{'val':200})
 cat=node(plot,C,'catAx');node(cat,C,'axId',{'val':100});sc=node(cat,C,'scaling');node(sc,C,'orientation',{'val':'maxMin' if horizontal else 'minMax'});node(cat,C,'delete',{'val':0});node(cat,C,'axPos',{'val':'l' if horizontal else 'b'});node(cat,C,'majorTickMark',{'val':'none'});node(cat,C,'minorTickMark',{'val':'none'});node(cat,C,'tickLblPos',{'val':'low'});sp=node(cat,C,'spPr');ln(sp,'C3CDD5');charttext(cat,'',s.get('font',11));node(cat,C,'crossAx',{'val':200});node(cat,C,'crosses',{'val':'autoZero'});node(cat,C,'auto',{'val':1});node(cat,C,'lblAlgn',{'val':'ctr'});node(cat,C,'lblOffset',{'val':100})
 if n>10:node(cat,C,'tickLblSkip',{'val':max(1,math.ceil(n/7))});node(cat,C,'tickMarkSkip',{'val':max(1,math.ceil(n/7))})
 va=node(plot,C,'valAx');node(va,C,'axId',{'val':200});sc=node(va,C,'scaling')
 if s.get('log'):node(sc,C,'logBase',{'val':10})
 node(sc,C,'orientation',{'val':'minMax'})
 for label in ['max','min']:
  if label in s:node(sc,C,label,{'val':s[label]})
 node(va,C,'delete',{'val':0});node(va,C,'axPos',{'val':'b' if horizontal else 'l'});grid=node(va,C,'majorGridlines');ln(node(grid,C,'spPr'),'DFE5EA',.5);node(va,C,'numFmt',{'formatCode':s.get('axis_fmt',fmt),'sourceLinked':0});node(va,C,'majorTickMark',{'val':'none'});node(va,C,'minorTickMark',{'val':'none'});node(va,C,'tickLblPos',{'val':'nextTo'});ln(node(va,C,'spPr'),'C3CDD5');charttext(va,'',s.get('font',11));node(va,C,'crossAx',{'val':100});node(va,C,'crosses',{'val':'max' if horizontal else 'autoZero'});node(va,C,'crossBetween',{'val':'between'});node(va,C,'majorUnit',{'val':(s['max']-s['min'])/5})
 if s.get('legend',len(series)>1):
  le=node(ch,C,'legend');node(le,C,'legendPos',{'val':'b'});node(le,C,'overlay',{'val':0});charttext(le,'',s.get('font',11))
 node(ch,C,'plotVisOnly',{'val':1});node(ch,C,'dispBlanksAs',{'val':'gap'});node(ch,C,'showDLblsOverMax',{'val':0});sp=node(r,C,'spPr');solid(sp,'FFFFFF');node(node(sp,A,'ln'),A,'noFill');charttext(r,'',s.get('font',11));ex=node(r,C,'externalData',{q(REL,'id'):'rId1'});node(ex,C,'autoUpdate',{'val':0});return xml(r),workbook(cats,series)
J=json.loads((R/'data/verified_experiment.json').read_text());SL=json.loads((R/'data/slides.json').read_text())
def df(v):return pd.DataFrame(v['data'],index=v['index'],columns=v['columns'])
T={k:df(v) for k,v in J['tables'].items()};TR={k:df(v) for k,v in J['traces'].items()};COL={'Baseline PPO':'2166AC','Selected PPO':'C66118','Cash':'777C85','Buy and hold':'237A57'}
def ser(name,v,color='2166AC',**kw):return {'name':name,'values':[float(x) if x is not None else None for x in v],'color':color,**kw}
chart_pages={}
def charts(slide,specs):chart_pages[slide]=specs
BASIC=(45,85,870,350)
half1=(45,85,425,350);half2=(490,85,425,350)
daily=df(J['daily_close']);dates=pd.to_datetime(daily.index,utc=True)
foldcats=[];starts=[];lens=[]
for i,(a,b) in enumerate(J['protocol']['folds']+[[J['protocol']['evaluation_start'],J['protocol']['evaluation_end']]]):
 a,b=pd.Timestamp(a),pd.Timestamp(b);foldcats.append(('Evaluation' if i==4 else f'Validation {i+1}')+f'  {a:%d %b}–{b:%d %b}');starts.append((a-pd.Timestamp('2026-01-01',tz='UTC')).days);lens.append((b-a).days)
charts(4,[({'title':'Daily BTC/USDT close (logarithmic scale)','kind':'line','categories':dates.strftime('%b %Y').tolist(),'series':[ser('BTC/USDT',daily.close)],'log':True,'axis_fmt':'#,##0','legend':False,'font':10},(45,85,870,198)),({'title':'2026 windows: days since 1 January; endpoints exclusive','categories':foldcats,'series':[ser('Start offset',starts,hidden=True),ser('Duration',lens,colors=['C66118']*4+['237A57'])],'horizontal':True,'grouping':'stacked','legend':False,'axis_fmt':'0','min':0,'max':273,'font':10,'title_size':12},(45,285,870,160))])
pos=J['protocol']['configurations'][0]['positions'];charts(6,[({'title':'Nine discrete target positions','categories':[str(p) for p in pos],'series':[ser('Exposure',pos,colors=['AE4E56' if p<0 else '777C85' if p==0 else '237A57' for p in pos])],'fmt':'0.##','min':-1.1,'max':1.1,'labels':True,'legend':False},BASIC)])
s=T['screening_summary'].sort_values('mean_return');raw=T['screening_trials'];series=[ser('Four-fold mean',s.mean_return,'9EBACB')]
for fold in range(4):series.append(ser(f'Fold {fold+1}',[raw.loc[(raw.config==k)&(raw.fold==fold),'portfolio_return'].iloc[0] for k in s.index],'142735',kind='line',marker=True,no_line=True))
charts(11,[({'title':'Screening: fold returns and four-fold means','categories':[k.replace('ppo_','').replace('_','\n') for k in s.index],'series':series,'fmt':'0.0%','axis_fmt':'0%','font':11},BASIC)])
keys=['ppo_baseline','ppo_momentum','ppo_low_turnover'];charts(12,[({'title':'Mean individual-agent validation return','categories':['Baseline','Momentum','Low turnover'],'series':[ser('Screening: 1 seed × 4 folds',T['screening_summary'].loc[keys,'mean_return'],'9EBACB'),ser('Confirmation: 3 seeds × 4 folds',T['confirmation_individual_summary'].loc[keys,'mean_return'])],'fmt':'0.00%','axis_fmt':'0%','labels':True,'min':-.13,'max':.035},BASIC)])
s=T['ensemble_summary'].copy();s.loc['Cash']=0;s=s.sort_values('selection_score',ascending=False);charts(14,[({'title':'Validation selection score (higher is better)','categories':[k.replace('ppo_','').replace('::',' / ').replace('_',' ') for k in s.index],'series':[ser('Score',s.selection_score,colors=['777C85' if k=='Cash' else 'C66118' if k=='ppo_baseline::unanimous' else '2166AC' for k in s.index])],'horizontal':True,'fmt':'0.000000','axis_fmt':'0.00','labels':True,'min':-.12,'max':.02,'font':10},BASIC)])
v=T['validation_fold_comparison'];charts(15,[({'title':'Portfolio return by validation fold','categories':['19 Jan–18 Feb','20 Mar–19 Apr','19 May–18 Jun','18 Jul–17 Aug'],'series':[ser('Buy and hold',v['Buy and hold'],COL['Buy and hold']),ser('Baseline PPO',v['ppo_baseline::mean_probability']),ser('Selected PPO',v['ppo_baseline::unanimous'],COL['Selected PPO']),ser('Cash',v['Cash'],COL['Cash'],kind='line',dash=True)],'fmt':'0.0%','axis_fmt':'0%'},BASIC)])
co=T['policy_comparison'];order=list(COL);charts(16,[({'title':'Evaluation net return','categories':['Baseline\nPPO','Selected\nPPO','Cash','Buy and\nhold'],'series':[ser('Net return',co.loc[order,'portfolio_return'],colors=[COL[k] for k in order])],'fmt':'0.00%','axis_fmt':'0%','labels':True,'min':-.04,'max':.26,'font':11},half1),({'title':'Maximum drawdown magnitude','categories':['Baseline\nPPO','Selected\nPPO','Cash','Buy and\nhold'],'series':[ser('Drawdown',co.loc[order,'max_drawdown'].abs(),colors=[COL[k] for k in order])],'fmt':'0.00%','axis_fmt':'0%','labels':True,'min':0,'max':.10,'font':11},half2)])
cats=pd.to_datetime(TR['Baseline PPO'].timestamp).dt.strftime('%d %b\n%H:%M').tolist();rets=[];dds=[]
for k in order:
 v=TR[k].portfolio_value.to_numpy();rets.append(ser(k,v/v[0]-1,COL[k]));dds.append(ser(k,v/np.maximum.accumulate(v)-1,COL[k]))
charts(17,[({'title':'Cumulative return','kind':'line','categories':cats,'series':rets,'fmt':'0.0%','axis_fmt':'0%','font':10},(45,85,870,211)),({'title':'Drawdown','kind':'line','categories':cats,'series':dds,'fmt':'0.0%','axis_fmt':'0%','legend':False,'font':10,'max':.001},(45,300,870,145))])
charts(18,[({'title':'Share of 743 decisions at each requested position','categories':[str(p) for p in pos],'series':[ser(k,TR[k].position.iloc[1:].value_counts().reindex(pos,fill_value=0)/743,COL[k]) for k in ['Baseline PPO','Selected PPO']],'fmt':'0.00%','axis_fmt':'0%','labels':False,'min':0,'max':1.1},BASIC)])
s=T['final_seed_results'].loc[lambda d:d.policy=='Baseline PPO'];charts(19,[({'title':'Individual final seeds and ensemble references','categories':[f'Seed {k}' for k in s.seed],'series':[ser('Individual network',s.portfolio_return,colors=['AE4E56','779DAC','2166AC']),ser('Mean probability +14.44%',[co.loc['Baseline PPO','portfolio_return']]*3,COL['Baseline PPO'],kind='line',dash=True),ser('Unanimous −0.57%',[co.loc['Selected PPO','portfolio_return']]*3,COL['Selected PPO'],kind='line',dash=True)],'fmt':'0.00%','axis_fmt':'0%','labels':True,'min':-.20,'max':.24},BASIC)])
hist=[(ast.literal_eval(k)[2],df(v)) for k,v in J['histories'].items()];metrics=['entropy','value_loss','approx_kl','explained_variance'];titles=['Policy entropy (nats)','Critic half-MSE','Approximate KL','Critic explained variance'];specs=[]
for i,(m,title_) in enumerate(zip(metrics,titles)):
 specs.append(({'title':title_,'kind':'line','categories':[str(int(x/1024)) for x in hist[0][1].steps],'series':[ser(f'Seed {seed}',h[m],c) for (seed,h),c in zip(hist,['2166AC','C66118','237A57'])],'fmt':'0.000' if m=='approx_kl' else '0.00','font':9,'title_size':13,'legend':i==0},(45+(i%2)*445,85+(i//2)*180,425,175)))
charts(20,specs)
# Base package with a blank master/layout and explicit fonts.
parts={};chart_specs=[];chart_count=0
ref=Path('/Users/alivakof/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx')
with zipfile.ZipFile(ref) as z:parts['ppt/theme/theme1.xml']=z.read('ppt/theme/theme1.xml')
master=E.Element(q(P,'sldMaster'));group(node(master,P,'cSld'));node(master,P,'clrMap',{'bg1':'lt1','tx1':'dk1','bg2':'lt2','tx2':'dk2','accent1':'accent1','accent2':'accent2','accent3':'accent3','accent4':'accent4','accent5':'accent5','accent6':'accent6','hlink':'hlink','folHlink':'folHlink'});node(node(master,P,'sldLayoutIdLst'),P,'sldLayoutId',{'id':2147483649,q(REL,'id'):'rId1'});node(master,P,'txStyles');parts['ppt/slideMasters/slideMaster1.xml']=xml(master);parts['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')])
layout=E.Element(q(P,'sldLayout'),{'type':'blank','preserve':'1'});group(node(layout,P,'cSld',{'name':'Blank'}));node(node(layout,P,'clrMapOvr'),A,'masterClrMapping');parts['ppt/slideLayouts/slideLayout1.xml']=xml(layout);parts['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')])
nm=E.Element(q(P,'notesMaster'));group(node(nm,P,'cSld'));node(nm,P,'clrMap',{'bg1':'lt1','tx1':'dk1','bg2':'lt2','tx2':'dk2','accent1':'accent1','accent2':'accent2','accent3':'accent3','accent4':'accent4','accent5':'accent5','accent6':'accent6','hlink':'hlink','folHlink':'folHlink'});node(nm,P,'notesStyle');parts['ppt/notesMasters/notesMaster1.xml']=xml(nm);parts['ppt/notesMasters/_rels/notesMaster1.xml.rels']=rels([('rId1','theme','../theme/theme1.xml')])
pres=E.Element(q(P,'presentation'));node(node(pres,P,'sldMasterIdLst'),P,'sldMasterId',{'id':2147483648,q(REL,'id'):'rId1'});node(node(pres,P,'notesMasterIdLst'),P,'notesMasterId',{q(REL,'id'):'rId2'});ids=node(pres,P,'sldIdLst');presrels=[('rId1','slideMaster','slideMasters/slideMaster1.xml'),('rId2','notesMaster','notesMasters/notesMaster1.xml')]
pdf=pymupdf.open(R/'PPO_Presentation.pdf');table_pages=[]
for i,(page,meta) in enumerate(zip(pdf,SL),1):
 slide=Slide(i,'142735' if i==1 else 'FFFFFF');soup=BeautifulSoup(meta['body'],'html.parser');table=soup.find('table');tablebounds=None
 if table:
  rows=[tr.find_all(['th','td'],recursive=False) for tr in table.find_all('tr')];draws=page.get_drawings();borders=[d['rect'] for d in draws if d['fill'] and abs(d['rect'].height-.75)<.05 and abs(d['fill'][0]-.8353)<.001]
  cols=sorted(set(round(x,2) for r in borders for x in [r.x0,r.x1]));ends=sorted(set(round(r.y1,2) for r in borders));assert len(ends)==len(rows)
  top=81.75 if i!=7 else ends[0]-46.5;ys=[top]+ends;tablebounds=pymupdf.Rect(cols[0],top,cols[-1],ends[-1]);slide.table(rows,cols,ys);table_pages.append(i)
 # Preserve the original text positions and style from the verified PDF.
 for block in page.get_text('dict')['blocks']:
  if block['type']!=0:continue
  for line in block['lines']:
   rect=pymupdf.Rect(line['bbox'])
   if tablebounds and tablebounds.contains(rect):continue
   slide.textline(line)
 for d in page.get_drawings():
  rect=d['rect']
  if d['fill'] and rect.height<3 and rect.width>500:
   if tablebounds and tablebounds.contains(rect):continue
   slide.rect(rect,''.join(f'{round(c*255):02X}' for c in d['fill']))
 for spec,box in chart_pages.get(i,[]):slide.chart(spec,box)
 parts[f'ppt/slides/slide{i}.xml']=xml(slide.root);parts[f'ppt/slides/_rels/slide{i}.xml.rels']=rels(slide.rels)
 notes=E.Element(q(P,'notes'));st=group(node(notes,P,'cSld'));sp=node(st,P,'sp');nv=node(sp,P,'nvSpPr');node(nv,P,'cNvPr',{'id':2,'name':'Speaker notes'});node(nv,P,'cNvSpPr');node(node(nv,P,'nvPr'),P,'ph',{'type':'body','idx':1});node(sp,P,'spPr');tb=txbody(sp)
 for text in [meta['note'],'Source: Project_PPO_All_In_One_Fixed.ipynb; verified_experiment.json. '+meta['foot']]:para(tb,text,12)
 node(node(notes,P,'clrMapOvr'),A,'masterClrMapping');parts[f'ppt/notesSlides/notesSlide{i}.xml']=xml(notes);parts[f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels']=rels([('rId1','notesMaster','../notesMasters/notesMaster1.xml'),('rId2','slide',f'../slides/slide{i}.xml')]);node(ids,P,'sldId',{'id':255+i,q(REL,'id'):f'rId{i+2}'});presrels.append((f'rId{i+2}','slide',f'slides/slide{i}.xml'))
node(pres,P,'sldSz',{'cx':em(960),'cy':em(540),'type':'screen16x9'});node(pres,P,'notesSz',{'cx':6858000,'cy':9144000});parts['ppt/presentation.xml']=xml(pres);parts['ppt/_rels/presentation.xml.rels']=rels(presrels);parts['_rels/.rels']=rels([('rId1','officeDocument','ppt/presentation.xml')])
ct=E.Element(q(CT,'Types'));node(ct,CT,'Default',{'Extension':'rels','ContentType':'application/vnd.openxmlformats-package.relationships+xml'});node(ct,CT,'Default',{'Extension':'xml','ContentType':'application/xml'});node(ct,CT,'Default',{'Extension':'xlsx','ContentType':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'})
for path in parts:
 typ=None
 if path=='ppt/presentation.xml':typ='presentationml.presentation.main'
 elif re.match('ppt/slides/slide[0-9]+.xml$',path):typ='presentationml.slide'
 elif re.match('ppt/slideMasters/slideMaster[0-9]+.xml$',path):typ='presentationml.slideMaster'
 elif re.match('ppt/slideLayouts/slideLayout[0-9]+.xml$',path):typ='presentationml.slideLayout'
 elif re.match('ppt/notesSlides/notesSlide[0-9]+.xml$',path):typ='presentationml.notesSlide'
 elif re.match('ppt/notesMasters/notesMaster[0-9]+.xml$',path):typ='presentationml.notesMaster'
 elif re.match('ppt/theme/theme[0-9]+.xml$',path):typ='theme'
 elif re.match('ppt/charts/chart[0-9]+.xml$',path):typ='drawingml.chart'
 if typ:node(ct,CT,'Override',{'PartName':'/'+path,'ContentType':'application/vnd.openxmlformats-officedocument.'+typ+'+xml'})
parts['[Content_Types].xml']=xml(ct)
for k,v in parts.items():
 if k.endswith('.xml') or k.endswith('.rels'):E.fromstring(v)
path=B/'PPO_Presentation_draft.pptx'
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for k,v in parts.items():z.writestr(k,v)
(B/'native_chart_specs.json').write_text(json.dumps(chart_specs,indent=2));(B/'pptx_requirements.json').write_text(json.dumps({'slides':24,'native_chart_slides':sorted(chart_pages),'native_tables':table_pages,'charts':chart_count,'speaker_notes':24,'dimensions_emu':[em(960),em(540)],'font_policy':{'basis':'reference','families':['Arial','Times New Roman'],'referencePath':str(R/'PPO_Presentation.pdf')}},indent=2))
print('Created',path,'with',len(SL),'slides,',chart_count,'editable charts,',len(table_pages),'native tables and embedded speaker notes.')
