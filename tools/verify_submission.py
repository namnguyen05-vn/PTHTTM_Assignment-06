"""Check the final deliverables without training or changing saved model results."""
import argparse
import ast
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.io import read_json,write_json,sha256
from pypdf import PdfReader

def verify(visual_reviewed=False):
    pdf=ROOT/"report/Assignment06_NguyenNgocHoangNam_B23DCCN585.pdf"
    reader=PdfReader(pdf)
    assert len(reader.pages)==len(read_json(ROOT/"results/report_layout.json"))+1
    text="\n".join(page.extract_text() for page in reader.pages)
    for bad in ["S&P;","bản thảo trước thực nghiệm","chưa có kết quả","hãy chạy"]:
        assert bad not in text.lower() if bad==bad.lower() else bad not in text
    assert "CHƯƠNG 7. KẾT LUẬN" in text
    urls=[]
    for page in reader.pages:
        for annotation in page.get("/Annots",[]):
            action=annotation.get_object().get("/A",{})
            if action.get("/URI"):urls.append(str(action["/URI"]))
    for url in ["https://github.com/namnguyen05-vn/PTHTTM_Assignment-06",
                "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
                "https://www.kaggle.com/datasets/camnugent/sandp500"]:
        assert url in urls
    records=read_json(ROOT/"results/metric_verification.json")
    assert len(records)==4 and all(r["verified"] for r in records)
    provenance=read_json(ROOT/"results/provenance_verification.json")
    assert len(provenance)==4
    assert all(r["metadata_matches"] and r["history_and_selection_verified"] and r["csv_predictions_verified"] for r in provenance)
    runs=[]
    for d in ["retailrocket","sp500"]:
        for f in ["pytorch","keras"]:
            path=ROOT/"results/runs"/f"{d}_{f}_seed42"
            m=read_json(path/"metrics.json")
            assert not m["smoke"] and m["evaluation_split"]=="test"
            assert read_json(ROOT/f"results/checkpoint_{d}_{f}.json")["passed"]
            runs.append(dict(dataset=d,framework=f,samples=m["samples"],best_epoch=m["best_epoch"],completed_epochs=m["completed_epochs"]))
    notebooks=[]
    for path in sorted((ROOT/"notebooks").glob("*.ipynb")):
        n=read_json(path);cells=[c for c in n["cells"] if c["cell_type"]=="code"]
        for cell in cells:
            assert cell["execution_count"] is not None
            assert not any(o["output_type"]=="error" for o in cell["outputs"])
            source="".join(cell["source"])
            assert len(source.splitlines())<=35
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="RUN_FULL" for t in node.targets):
                    assert isinstance(node.value,ast.Constant) and node.value.value is False
        notebooks.append(dict(file=path.name,code_cells=len(cells),max_cell_lines=max(len("".join(c["source"]).splitlines()) for c in cells)))
    assert len(notebooks)==10
    manifest=[]
    for path in sorted((ROOT/"results/runs").rglob("*")):
        if path.is_file() and path.suffix!=".lock":
            manifest.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha256(path)))
    write_json(ROOT/"results/run_artifact_manifest.json",manifest)
    result=dict(status="complete_and_visually_reviewed" if visual_reviewed else "technical_checks_passed_visual_review_pending",
        full_training_runs=runs,notebooks=notebooks,pdf_pages=len(reader.pages),pdf_sha256=sha256(pdf),
        visual_review_completed=visual_reviewed,links_verified=True,run_artifact_files=len(manifest))
    write_json(ROOT/"results/final_verification.json",result)
    print(result["status"],"PDF pages:",len(reader.pages),"notebooks:",len(notebooks),"run files:",len(manifest))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--visual-reviewed",action="store_true")
    verify(parser.parse_args().visual_reviewed)
