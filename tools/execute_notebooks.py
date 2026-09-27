import asyncio
import ast
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.io import write_json
import nbformat
from nbclient import NotebookClient
if os.name=="nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
if __name__=="__main__":
    # Validate every notebook before executing any cell.
    for path in sorted((ROOT/"notebooks").glob("*.ipynb")):
        notebook=nbformat.read(path,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type!="code":
                continue
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node,ast.Assign) and any(
                    isinstance(t,ast.Name) and t.id=="RUN_FULL" for t in node.targets):
                    if not isinstance(node.value,ast.Constant) or node.value.value is not False:
                        raise SystemExit(f"Set RUN_FULL=False before report execution: {path.name}")
    records=[]
    for path in sorted((ROOT/"notebooks").glob("*.ipynb")):
        notebook=nbformat.read(path,as_version=4)
        client=NotebookClient(notebook,timeout=600,kernel_name="assignment06",
            resources={"metadata":{"path":str(ROOT/"notebooks")}})
        client.execute()
        nbformat.write(notebook,path)
        code=[c for c in notebook.cells if c.cell_type=="code"]
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type=="error" for c in code for o in c.outputs)
        records.append(dict(notebook=path.name,code_cells=len(code),
            max_cell_lines=max(len(c.source.splitlines()) for c in code),passed=True))
        print("Executed",path.name,flush=True)
    write_json(ROOT/"results/notebook_verification.json",records)
