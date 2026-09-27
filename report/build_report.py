"""Draft excludes unrun experiments; final mode requires verified full output."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from report_engine import PAGES,render
from cover import create_cover
from theory import theory
from content import front,introduction,datasets,implementation,methods,finish,indexes

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--draft",action="store_true")
    args=parser.parse_args()
    create_cover()
    front(args.draft);introduction();theory();datasets();implementation();methods()
    if not args.draft:
        from results import result_pages
        result_pages()
    finish(args.draft);indexes()
    name="Assignment06_BanThao_TruocHu anLuyen.pdf".replace(" ","") if args.draft else "Assignment06_NguyenNgocHoangNam_B23DCCN585.pdf"
    render(ROOT/"report"/name,expected_pages=len(PAGES)+1)
