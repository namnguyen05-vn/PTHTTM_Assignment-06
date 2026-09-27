"""Small cells for concepts, data, framework implementations and result inspection."""
import ast
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
import nbformat as nb
M=nb.v4.new_markdown_cell
C=nb.v4.new_code_cell
SETUP="""from pathlib import Path
import sys, json, os
ROOT=Path.cwd().resolve()
if ROOT.name=="notebooks": ROOT=ROOT.parent
assert (ROOT/"src").is_dir()
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.paths import DATA
from src.io import read_json
import numpy as np
import pandas as pd
from IPython.display import display, Image
print("Project:", ROOT)
print("Data:", DATA)
"""
def source_function(module,name):
    text=(ROOT/"src"/module).read_text(encoding="utf-8-sig")
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return "\n".join(text.splitlines()[node.lineno-1:node.end_lineno])
def write(name,title,cells):
    notebook=nb.v4.new_notebook(cells=[M("# "+title+"\n\nNguyễn Ngọc Hoàng Nam — B23DCCN585 | ASG 06"),
        M("Các cell nhỏ được sắp theo thứ tự phụ thuộc. Huấn luyện đầy đủ mặc định tắt; số liệu chạy thử không phải kết quả test chính thức."),
        C(SETUP)]+cells)
    notebook.metadata.kernelspec=dict(name="assignment06",display_name="Python (Assignment 06)",language="python")
    for cell in notebook.cells:
        if cell.cell_type=="code":
            ast.parse(cell.source)
            assert len(cell.source.splitlines())<=35,(name,len(cell.source.splitlines()))
    nb.write(notebook,ROOT/"notebooks"/(name+".ipynb"))
def build():
    write("00_RNN_Functions","RNN: hàm, trạng thái ẩn và BPTT",[
        M("## 1. Phép truy hồi\n$h_t=\\tanh(x_tW_x+h_{t-1}W_h+b)$. Cùng bộ trọng số được dùng ở mọi bước thời gian."),
        C(source_function("teaching.py","rnn_forward")),
        C("x=np.array([[1.,0.],[0.,1.],[1.,1.]])\nwx=np.array([[.3,-.2],[.1,.4]])\nwh=np.eye(2)*.5\nb=np.zeros(2)\nstates=rnn_forward(x,wx,wh,b)\ndisplay(pd.DataFrame(states,columns=['h1','h2']))"),
        M("## 2. Thứ tự có ý nghĩa\nĐảo thứ tự đầu vào có thể đổi trạng thái cuối dù tập giá trị không đổi."),
        C("print('Ban đầu:',states[-1])\nprint('Đảo thứ tự:',rnn_forward(x[::-1],wx,wh,b)[-1])"),
        M("## 3. Lan truyền ngược qua thời gian\nGradient của trọng số dùng chung là tổng đóng góp của từng bước."),
        C(source_function("teaching.py","last_state_gradient")),
        C(source_function("teaching.py","finite_difference")),
        C("target=np.array([.2,.1])\nloss,grads=last_state_gradient(x,wx,wh,b,target)\nnumeric=finite_difference(lambda a:last_state_gradient(x,a,wh,b,target)[0],wx)\nnp.testing.assert_allclose(grads[0],numeric,atol=1e-7)\nprint('Loss:',loss,'Max gradient error:',np.abs(grads[0]-numeric).max())"),
        M("## 4. Độ dài chuỗi và gradient\nTích nhiều hệ số nhỏ có thể làm gradient tiêu biến; gradient clipping chỉ giới hạn gradient quá lớn."),
        C("steps=np.arange(1,31)\ndisplay(pd.DataFrame({'steps':steps,'0.5^steps':.5**steps,'1.5^steps':1.5**steps}))")])
    for number,name in [(1,"retailrocket"),(2,"sp500")]:
        cells=[M("## 1. Nguồn và kiểm tra dữ liệu"),C(f"from src.data import load_data,split_ids,make_batch\ndata=load_data('{name}')\nmeta=data['metadata']\ndisplay(meta)"),
            M("## 2. Tập chia theo thời gian"),C("rows=[]\nfor split in ['train','val','test']:\n    ids=split_ids(data,split)\n    rows.append([split,len(ids),pd.to_datetime(data['times'][ids].min(),unit='s'),pd.to_datetime(data['times'][ids].max(),unit='s')])\ndisplay(pd.DataFrame(rows,columns=['split','samples','first target','last target']))"),
            M("## 3. Cấu trúc batch và phần đệm"),C("ids=split_ids(data,'train')[:8]\nx,lengths,y=make_batch(data,ids)\nprint('X:',x.shape,'lengths:',lengths,'targets:',y)\ndisplay(pd.DataFrame(x[0,:lengths[0]],columns=meta['features']))")]
        figures=["retail_distribution","retail_daily","retail_transition"] if name=="retailrocket" else ["stock_exploration","stock_windows"]
        for figure in figures:cells+=[C(f"display(Image(filename=str(ROOT/'figures/{figure}.png')))")]
        cells+=[M("## 4. Giới hạn\n"+("Chỉ dự đoán loại hành vi tiếp theo khi phiên còn sự kiện. Khách trở lại có thể xuất hiện ở nhiều giai đoạn. Sự kiện trùng thời điểm gây thứ tự không xác định bị loại và được thống kê." if name=="retailrocket" else "Dự đoán một phiên quan sát tiếp theo, dùng thông tin đến phiên trước. Dữ liệu lịch sử, thành phần chỉ số và cách điều chỉnh giá giới hạn khả năng suy rộng."))]
        write(f"{number:02d}_Data_{name}","Phân tích dữ liệu "+name,cells)
    idx=3
    for name in ["retailrocket","sp500"]:
        for framework in ["pytorch","keras"]:
            cells=[M("## 1. Dữ liệu và hình dạng"),C(f"from src.data import load_data,split_ids,make_batch\ndata=load_data('{name}')\nx,lengths,y=make_batch(data,split_ids(data,'train')[:8])\nprint(x.shape,lengths,y)"),
                M("## 2. Hàm xây dựng RNN cơ bản")]
            if framework=="pytorch":
                cells+=[C("from src.backends import pytorch_runtime\ntorch,device=pytorch_runtime(42,cpu=True)\nprint(torch.__version__,device)"),
                    C(source_function("backends.py","build_pytorch")),
                    C(f"model=build_pytorch(torch,data['metadata']['input_dim'],32,{3 if name=='retailrocket' else 1})\nprint(model)"),
                    M("## 3. Lan truyền thuận và hàm mất mát"),
                    C("tensor=torch.tensor(x)\noutput=model(tensor,torch.tensor(lengths))\nprint('Output:',output.shape)"),
                    C(("loss=torch.nn.functional.cross_entropy(output,torch.tensor(y,dtype=torch.long))" if name=="retailrocket" else "target=torch.tensor(y/data['metadata']['target_scale'],dtype=torch.float32)\nloss=torch.mean((output[:,0]-target)**2)")+"\nloss.backward()\nprint('Loss minh họa:',float(loss))\nprint('Norm gradient:',float(model.rnn.weight_ih_l0.grad.norm()))")]
            else:
                cells+=[C("from src.backends import keras_runtime\ntf=keras_runtime(42,cpu=True)\nprint(tf.__version__)"),
                    C(source_function("backends.py","build_keras")),
                    C(f"model=build_keras(tf,data['metadata']['input_dim'],32,{3 if name=='retailrocket' else 1},data['metadata']['sequence_length'])\nmodel.summary()"),
                    M("## 3. Lan truyền thuận và hàm mất mát"),
                    C("output=model([x,lengths],training=False)\nprint('Output:',output.shape)"),
                    C("with tf.GradientTape() as tape:\n    output=model([x,lengths],training=True)\n"+("    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,3),logits=output))" if name=="retailrocket" else "    target=tf.constant(y/data['metadata']['target_scale'],dtype=tf.float32)\n    loss=tf.reduce_mean(tf.square(output[:,0]-target))")+"\ngradients=tape.gradient(loss,model.trainable_variables)\nprint('Loss minh họa:',float(loss))\nprint('Norm gradient:',float(tf.linalg.global_norm(gradients)))")]
            cells+=[M("## 4. Bộ huấn luyện đầy đủ\nVí dụ trên giải thích đồ thị. Bộ huấn luyện dùng cùng họ kiến trúc, khởi tạo chung giữa framework, trọng số lớp cho Retailrocket, clipping và checkpoint. Mã thuật toán dùng cho lượt chính nằm trong src/backends.py và src/training.py."),
                C("from src.training import configuration\nconfig,_=configuration('"+name+"','"+framework+"')\ndisplay(config)"),
                M("## 5. Chủ động chạy cấu hình này\nChỉ chuyển RUN_FULL=True khi muốn huấn luyện đầy đủ. Nên dùng RUN_TRAINING.bat để giảm bộ nhớ của kernel minh họa đang mở."),
                C("RUN_FULL=False\nif RUN_FULL:\n    import subprocess\n    environment=os.environ.copy()\n    environment.pop('CUDA_VISIBLE_DEVICES',None)\n    subprocess.run([sys.executable,'-m','src.training','--dataset','"+name+"','--framework','"+framework+"'],cwd=ROOT,env=environment,check=True)\nelse:\n    print('Cell này không khởi động huấn luyện; kết quả đã lưu được đọc ở phần sau.')"),
                M("## 6. Dự đoán và kết quả được lưu"),
                C("folder=ROOT/'results/runs/"+name+"_"+framework+"_seed42'\nif (folder/'metrics.json').exists():\n    display(read_json(folder/'metrics.json'))\n    display(pd.read_csv(folder/'predictions.csv').head(12))\nelse:\n    print('Kết quả chính sẽ có sau khi bạn huấn luyện xong.')")]
            write(f"{idx:02d}_{framework}_{name}",framework.upper()+" RNN trên "+name,cells);idx+=1
    write("07_Training_and_Resume","Quy trình huấn luyện và tiếp tục checkpoint",[
        M("Bốn cấu hình được chạy lần lượt. PAUSE_TRAINING.bat yêu cầu dừng sau epoch hiện tại. Sau khi cửa sổ báo đã lưu và dừng, có thể Sleep. RUN_TRAINING.bat tiếp tục từ epoch gần nhất."),
        C("display(read_json(ROOT/'configs/main.json'))"),
        M("## Trạng thái thực nghiệm"),
        C("rows=[]\nfor name in ['retailrocket','sp500']:\n    for framework in ['pytorch','keras']:\n        folder=ROOT/'results/runs'/f'{name}_{framework}_seed42'\n        state=read_json(folder/'state.json') if (folder/'state.json').exists() else {}\n        rows.append([name,framework,state.get('epoch',0),(folder/'metrics.json').exists()])\ndisplay(pd.DataFrame(rows,columns=['dataset','framework','saved epoch','completed']))"),
        M("## Bằng chứng chạy tiếp\nSo sánh chạy liền hai epoch với dừng sau epoch một rồi khởi động lại; dữ liệu chỉ là smoke subset."),
        C("path=ROOT/'results/resume_verification.json'\nif path.exists(): display(read_json(path))")])
    write("08_Comparison_and_Visualization","Đánh giá, so sánh và trực quan hóa",[
        M("Chỉ kết quả trong results/runs được dùng cho báo cáo chính. Smoke được lưu riêng. Các đối chứng được fit trên train."),
        C("records=[]\nfor folder in sorted((ROOT/'results/runs').glob('*')):\n    if (folder/'metrics.json').exists(): records.append(read_json(folder/'metrics.json'))\nif records: display(pd.DataFrame(records))\nelse: print('Chưa có kết quả huấn luyện đầy đủ.')"),
        C("path=ROOT/'results/baselines.json'\nif path.exists(): display(pd.DataFrame(read_json(path)))"),
        C("for name in ['retailrocket_learning','sp500_learning','retailrocket_pytorch_confusion','retailrocket_keras_confusion','sp500_pytorch_forecast','sp500_keras_forecast','retail_pr','stock_diagnostics','stock_monthly']:\n    path=ROOT/'figures'/(name+'.png')\n    if path.exists(): display(Image(filename=str(path)))"),
        C("path=ROOT/'results/paired_framework_analysis.json'\nif path.exists(): display(read_json(path))"),
        M("Kết quả thực nghiệm: macro-F1 Retailrocket xấp xỉ 0,582, cao hơn đối chứng argmax luôn-view; accuracy và log loss không vượt mọi đối chứng. RNN cổ phiếu có RMSE khoảng 0,015533, cao hơn đối chứng trung bình train 0,015291. Cả hai framework chọn epoch 8 cho Retailrocket và epoch 1 cho S&P 500."),
        M("Retailrocket: macro-F1 và từng lớp bổ sung accuracy trong tình huống mất cân bằng. Chứng khoán: so sánh sai số log return với dự đoán zero-return; giá dự báo sát giá thực không tự chứng minh có giá trị vượt đối chứng.")])
    write("09_Verification","Hồ sơ kiểm chứng và tái lập",[
        M("Kiểm chứng dữ liệu, NumPy, masking, optimizer và khởi động lại tiến trình bổ sung cho đánh giá dự đoán."),
        C("for name in ['data_verification','verification_pytorch','verification_keras','resume_verification','provenance_verification','checkpoint_retailrocket_pytorch','checkpoint_retailrocket_keras','checkpoint_sp500_pytorch','checkpoint_sp500_keras']:\n    path=ROOT/'results'/(name+'.json')\n    if path.exists(): display(read_json(path))"),
        M("## Kiểm chứng kết quả sau huấn luyện"),
        C("path=ROOT/'results/metric_verification.json'\nif path.exists(): display(read_json(path))\nelse: print('Sẽ kiểm chứng metric sau khi đủ bốn lượt chính.')")])
    print("Built",len(list((ROOT/'notebooks').glob('*.ipynb'))),"notebooks")
if __name__=="__main__":build()
