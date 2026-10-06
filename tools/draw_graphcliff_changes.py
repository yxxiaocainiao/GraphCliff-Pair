"""Six source-grounded teacher diagrams; no model imports or inference."""
from pathlib import Path
import json,hashlib,warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures/graphcliff_changes_20261006'
FONT=Path('C:/Windows/Fonts/msyh.ttc');assert FONT.exists();font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=str(FONT)).get_name(),'DejaVu Sans'],'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
BLUE='#29628E';ORANGE='#B65B12';GREY='#78848D';INK='#253546';AX=None

def text(x,y,s,size=16,color=INK,ha='center',weight='normal'):
    return AX.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,fontweight=weight)
def box(x,y,w,h,s,new=False,grey=False,size=16,dashed=False):
    c=GREY if grey else ORANGE if new else BLUE;f='#F1F2F3' if grey else '#FFF0DE' if new else '#EEF5FC'
    AX.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.13',linewidth=1.8,edgecolor=c,facecolor=f,linestyle='--' if dashed else '-'))
    text(x+w/2,y+h/2,s,size,c);return (x,y,w,h)
def arrow(x,y,u,v,new=False,dashed=False):
    AX.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=17,lw=1.8,color=ORANGE if new else BLUE,linestyle='--' if dashed else '-',connectionstyle='arc3'))
def line(points,new=False,dashed=False):
    AX.plot(*zip(*points),color=ORANGE if new else BLUE,lw=1.7,linestyle='--' if dashed else '-')
def circle(x,y,s):
    AX.add_patch(Circle((x,y),.23,facecolor='white',edgecolor=BLUE,lw=1.7));text(x,y,s,17)
def newfig(title,subtitle,foot):
    global AX
    fig,AX=plt.subplots(figsize=(17,9));AX.set_xlim(0,20);AX.set_ylim(0,10);AX.axis('off');fig.subplots_adjust(left=.015,right=.985,top=.985,bottom=.015)
    text(.3,9.55,title,24,ha='left',weight='bold');text(.3,8.94,subtitle,15,ha='left')
    text(.3,.68,foot,14,ha='left');text(.3,.22,'颜色：蓝＝沿用   橙＝新增/替换   灰＋删除线＝被替换   灰虚线＝未定方案',13,ha='left',color=GREY)
    return fig

def baseline():
    f=newfig('01｜原GraphCliff：改动位置的参照图','按用户提供框架重绘；补画代码中的层输入残差，省略维度与部分归一化。','图读出位置＝Pool；层内长程位置＝LongPoly；Loss在训练输出之后，并非Filter内部。')
    box(.4,4.3,1.25,1.15,'分子图');box(2,4.3,1.6,1.15,'Atom\nEncoder');arrow(1.7,4.88,1.95,4.88);arrow(3.65,4.88,4.6,4.88)
    AX.add_patch(FancyBboxPatch((4.25,2.5),9.6,5.55,boxstyle='round,pad=.05,rounding_size=.3',edgecolor=GREY,facecolor='#FAFCFE',lw=1.5,linestyle='--'))
    text(9.05,7.62,'GraphCliff Filter × L',19,weight='bold')
    box(4.65,4.25,1.5,1.25,'Norm\nProj W');box(6.6,4.25,1.65,1.25,'SHORT\nGINE');arrow(6.2,4.88,6.55,4.88);arrow(8.3,4.88,8.7,4.88)
    line([(8.75,3.2),(8.75,6.4)]);text(9.05,6.4,'x₁',18);text(9.05,4.88,'x₂',18);text(9.05,3.2,'v',18)
    box(9.7,5.9,1.6,1,'Sigmoid σ');box(9.7,4.3,1.65,1.15,'LONG\nChebyshev',size=15);arrow(9.3,6.4,9.65,6.4);arrow(9.3,4.88,9.65,4.88);arrow(11.4,4.88,11.9,4.88)
    circle(12.15,4.88,'×');line([(11.35,6.4),(12.15,6.4)]);arrow(12.15,6.4,12.15,5.15);circle(13.1,4.88,'+');arrow(12.4,4.88,12.82,4.88)
    line([(9.3,3.2),(13.1,3.2)]);arrow(13.1,3.2,13.1,4.6);line([(4.4,4.88),(4.4,7.15),(13.1,7.15)]);arrow(13.1,7.15,13.1,5.15);text(6.3,7.36,'层输入 h（残差）',13)
    arrow(13.38,4.88,14.1,4.88);box(14.15,4.1,2.2,1.55,'Pool\nSAG + Max/Mean',size=15);arrow(16.4,4.88,16.8,4.88);box(16.85,4.1,1.7,1.55,'Regression');arrow(18.6,4.88,19.2,4.88);text(19.5,4.88,'ŷ',22)
    box(5.1,1.28,7.6,.65,'代码：h′ = h + σ(x₁) ⊙ LongPoly(x₂) + v',size=16)
    return f

def fppool():
    f=newfig('02｜FPPool：改Pool，不改GraphCliff Filter','直接替换与残差分支是两个已试方案；不是同一个模型同时采用两种读出。','FPPool使用编码节点与结构指纹归属；残差分支在SAG前读完整节点。两方案均未建立通用优势。')
    text(.4,8.15,'A 直接替换',18,ha='left',weight='bold');box(.5,6.55,1.4,1,'分子图');box(2.5,6.55,4.7,1,'Atom + GraphCliff Filter × L');arrow(1.95,7.05,2.45,7.05)
    box(8.15,6.55,3.15,1,'FPPool + Projection',new=True);arrow(7.25,7.05,8.1,7.05,new=True);box(12.6,6.55,2.5,1,'原Regression');arrow(11.35,7.05,12.55,7.05,new=True);arrow(15.15,7.05,17,7.05);text(17.6,7.05,'ŷ',22)
    box(8.15,7.9,3.15,.6,'原 SAG + Max/Mean',grey=True,size=14);line([(8.2,7.95),(11.25,8.45)],new=True);arrow(9.73,7.85,9.73,7.62,new=True)
    box(8.15,5.05,3.15,.65,'Morgan 原子－指纹归属',new=True,size=14);arrow(9.73,5.75,9.73,6.5,new=True)
    text(.4,4.5,'B 残差增加',18,ha='left',weight='bold');box(.5,2.7,1.4,1,'分子图');box(2.5,2.7,4.7,1,'Atom + GraphCliff Filter × L');arrow(1.95,3.2,2.45,3.2)
    box(8.2,3.5,3.15,1,'原 SAG + Max/Mean');box(8.2,1.7,3.15,1,'FPPool + Projection',new=True);line([(7.25,3.2),(7.65,3.2),(7.65,4)]);arrow(7.65,4,8.15,4);line([(7.65,3.2),(7.65,2.2)]);arrow(7.65,2.2,8.15,2.2,new=True)
    box(12.2,2.7,2.1,1,'g原 + α·gFP',new=True,size=16);line([(11.4,4),(13.25,4)]);arrow(13.25,4,13.25,3.75);line([(11.4,2.2),(13.25,2.2)],new=True);arrow(13.25,2.2,13.25,2.65,new=True);box(15.3,2.7,2.5,1,'原Regression');arrow(14.35,3.2,15.25,3.2,new=True);arrow(17.85,3.2,18.5,3.2);text(19,3.2,'ŷ',22)
    text(10,1.24,'α为可训练标量；未增加辅助Loss',14,color=ORANGE)
    return f

def post_cross():
    f=newfig('03｜编码后Cross-Attention：在Filter之后、Pool之前加交互','保留原ShortGINE + LongPoly双通道；增加查询与合法训练参考两个输入。','此路线没有替换双通道。预测先输出差值，再加训练参考活性恢复查询值；已停止扩展。')
    for y,name in [(6.4,'查询分子 Q'),(3.5,'训练参考 R')]:
        box(.45,y,2,1,name);box(3.1,y,4.4,1,'共享 AtomEncoder\n+ GraphCliff Filter × L');arrow(2.5,y+.5,3.05,y+.5)
        box(11.05,y,2.35,1,'原 SAG读出');box(7.9,y,2.5,1,'Cross-Attention',new=True);arrow(7.55,y+.5,7.85,y+.5,new=True);arrow(10.45,y+.5,11,y+.5,new=True)
    arrow(9.15,6.35,9.15,4.55,new=True);arrow(9.5,4.55,9.5,6.35,new=True);text(10.2,5.45,'原子表示\n双向交互',14,color=ORANGE)
    box(14.3,4.9,2.6,1.25,'gQ − gR\n反对称差值头',new=True);line([(13.45,6.9),(15.6,6.9)]);arrow(15.6,6.9,15.6,6.2,new=True);line([(13.45,4),(15.6,4)]);arrow(15.6,4,15.6,4.85,new=True);arrow(16.95,5.52,17.5,5.52,new=True);text(18.4,5.52,'Δŷ(Q,R)',19)
    box(5.35,1.6,12.5,1.05,'恢复查询：ŷQ = yR + Δŷ(Q,R)    ｜    yR仅来自合法训练参考',new=True,size=18)
    text(.55,7.98,'交互位置',15,color=ORANGE,ha='left');text(5.3,7.98,'编码器沿用',15);text(9.15,7.98,'新增',15,color=ORANGE);text(12.2,7.98,'读出沿用',15)
    return f

def branch_cross():
    f=newfig('04｜逐层Cross-Attention：只替换Filter里的LongPoly','每层Q/R同步：ShortGINE → 拆成x₁、x₂、v → x₂跨分子交互 → 原门控/残差融合。','这是另一组单seed替换试验；保留SHORT、门控与读出，还引入交互归一化等变化，非纯注意力消融。')
    for y,name in [(6.3,'查询 Q'),(3.1,'训练参考 R')]:
        box(.4,y,1.6,1,name);box(2.55,y,3.2,1,'Norm/Proj → SHORT');arrow(2.05,y+.5,2.5,y+.5);arrow(5.8,y+.5,6.3,y+.5);text(6.85,y+.5,'x₂',19)
        box(7.6,y,3.3,1,'CrossInteraction(x₂)',new=True);arrow(7.2,y+.5,7.55,y+.5,new=True);box(12.25,y,3.65,1,'h + σ(x₁)·x₂′ + v');arrow(10.95,y+.5,12.2,y+.5,new=True);arrow(15.95,y+.5,16.55,y+.5);box(16.6,y,2.35,1,'下一层 / Pool',size=15)
        line([(5.55,y+.98),(5.55,y+1.5),(14.08,y+1.5)]);arrow(14.08,y+1.5,14.08,y+1.06);text(12,y+1.75,'x₁、v与层输入h仍保留',14)
    arrow(9.0,6.25,9.0,4.15,new=True);arrow(9.4,4.15,9.4,6.25,new=True)
    box(1,4.82,4.35,.55,'被替换：LongPoly / Chebyshev',grey=True,size=14);line([(1.1,4.85),(5.28,5.34)],new=True)
    text(9.5,1.9,'输入为AtomEncoder后的原子表示；按L层重复，最后用差值头恢复查询活性。',15)
    return f

def losses():
    f=newfig('05｜Loss改动：训练端增加监督，不是改Filter结构','以下是三类独立实验；并未把全部Loss同时叠进同一最终模型。','橙色标签/损失分支仅训练时使用。动态＝epoch调度；不是预测误差自适应或元学习权重。')
    rows=[(6.85,'A 动态差值Loss','Pair模型 → Δŷ','训练真值差 Δy','加权差值 MSE','w ∝ 1 + α(epoch)·min(|Δy|/scale, cap)'),(4.55,'B 组件辅助Loss','单分子模型 → ŷ','训练分子对 (i,j)','MSE + λ·L差值','比较：balanced / uniform / permuted'),(2.25,'C ACA迁移','分子表示 + 活性预测','训练活性与三元组','MSE + α·LACA','复用ACANet损失；本地主任务为MSE')]
    for y,label,model,truth,loss,note in rows:
        text(.5,y+1.35,label,17,ha='left',weight='bold');box(.5,y,4,1,model);box(5.6,y,4.1,1,truth,new=True);box(11.05,y,4.1,1,loss,new=True);line([(4.55,y+.5),(4.9,y+.5),(4.9,y+1.15),(13.1,y+1.15)]);arrow(13.1,y+1.15,13.1,y+1.05);arrow(9.75,y+.5,11,y+.5,new=True);text(17.35,y+.5,'训练参数更新',15,color=ORANGE);arrow(15.2,y+.5,16.05,y+.5,new=True);text(10,y-.43,note,14,color=ORANGE)
    return f

def reliability():
    f=newfig('06｜后期方向：固定Chemprop，改预测可靠性','这不是在GraphCliff Filter中再加一层；主干已转为成熟Chemprop，点预测不随风险方法改变。','组件均衡已停止；图中灰虚线是尚未确定的新核心。已有风险收益不能直接称活性预测涨点或算法创新。')
    box(.45,6.35,1.55,1.15,'查询结构');box(2.65,6.35,3.15,1.15,'Chemprop D-MPNN\n固定点预测 ŷ',size=16);arrow(2.05,6.93,2.6,6.93)
    box(6.75,6.35,4.1,1.15,'通用5特征 + 粗糙度2特征',new=True,size=16);arrow(5.85,6.93,6.7,6.93,new=True);box(11.75,6.35,2.8,1.15,'风险RF → r(x)',new=True);arrow(10.9,6.93,11.7,6.93,new=True);box(15.5,6.35,3.95,1.15,'低风险优先接受\n六接受率RMSE',size=16);arrow(14.6,6.93,15.45,6.93)
    box(6.75,4.4,4.1,1,'合法训练近邻：结构 + 活性',new=True,size=15);arrow(8.8,5.45,8.8,6.3,new=True)
    box(11.75,4.4,2.8,1,'OOF绝对误差\n仅训练监督',new=True,size=15);arrow(13.15,5.45,13.15,6.3,new=True,dashed=True)
    text(.7,4.9,'查询真实活性不进推理特征',15,color=ORANGE,ha='left')
    box(.65,1.7,4.8,1.25,'已试：组件均衡粗糙度\n等组件离散度 / SALI块',new=True,size=16);box(6.25,1.7,5.1,1.25,'公平去核心：原粗糙度\n相同支持量、9维RF',size=16);text(5.86,2.32,'vs',18);box(12.4,1.7,6.6,1.25,'下一算法核心：未确定\n先定义问题与差异，当前不训练',grey=True,dashed=True,size=17)
    return f

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);sources=json.loads((OUT/'sources.json').read_text())
    assert all(hashlib.sha256((Path(s) if Path(s).is_absolute() else ROOT/s).read_bytes()).hexdigest()==h for s,h in sources['sources'].items())
    names=['01_original','02_fppool','03_post_cross','04_branch_cross','05_losses','06_reliability'];warnings.simplefilter('error',UserWarning)
    with PdfPages(OUT/'all_diagrams.pdf') as pdf:
        for name,draw in zip(names,[baseline,fppool,post_cross,branch_cross,losses,reliability]):
            fig=draw();fig.canvas.draw()
            # ponytail: bounds check prevents cropped labels; no general diagram engine needed for six fixed figures.
            renderer=fig.canvas.get_renderer();bound=fig.bbox
            for label in AX.texts:
                b=label.get_window_extent(renderer);assert b.x0>=bound.x0 and b.y0>=bound.y0 and b.x1<=bound.x1 and b.y1<=bound.y1,(name,label.get_text())
            for ext in ['png','svg','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=300,facecolor='white')
            svg=OUT/(name+'.svg');svg.write_text('\n'.join(row.rstrip() for row in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
            pdf.savefig(fig);plt.close(fig);print('DRAWN',name,flush=True)
    assert len(list(OUT.glob('*.png')))==6 and all((OUT/(n+'.'+e)).stat().st_size>1000 for n in names for e in ['png','svg','pdf'])
    (OUT/'gallery.html').write_text('<meta charset="utf-8"><title>GraphCliff改动图</title><style>body{max-width:1400px;margin:20px auto;font-family:sans-serif}img{width:100%}</style>'+''.join('<h2>'+n+'</h2><img src="'+n+'.png">' for n in names),encoding='utf-8')
    print('PASS: source bindings, label bounds, six figures × three formats, multipage PDF')
