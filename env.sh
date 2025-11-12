#!/usr/bin/bash
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
doi() {
  [ -z "$1" ] && echo -e "\x1b[33mdoi \x1b[0;2m{doi code}\x1b[0m\n  Returns the Bibtex of the DOI" && return
  curl -LH "Accept: application/x-bibtex" "https://doi.org/$1"
}
rst() {
  rm "$src/paper/build/"*
}
pdf() {
  [ "$1" = '-h' ] && echo -e "\x1b[33mpdf\x1b[0m\n  Compiles the Thesis Paper as PDF" && return
  #[ -e "$src/paper/build" ] && rm "$src/paper/build/"* -r
  mkdir -p "$src/paper/build"
  cd "$src/paper/build"
  cp ../*.{tex,bib} ../{pages,assets}/* .
  lualatex -shell-escape thesis.tex
  [ -e "$src/paper/build/thesis.pdf" ] && cp "$src/paper/build/thesis.pdf" ..
  mv *.svg ../assets
  rm *.tex *.ttf *.bib svg-inkscape -r
  cd -
}
liv() {
  cd "$src/paper/build"
  cp ../*.{tex,bib} ../{pages,assets}/* .
  latexmk -lualatex -synctex=1 -pvc thesis.tex
  #latexmk -lualatex -pvc -interaction=nonstopmode thesis.tex
  [ -e "$src/paper/build/thesis.pdf" ] && cp "$src/paper/build/thesis.pdf" ..
  mv *.svg ../assets
  rm *.tex *.ttf *.bib svg-inkscape -r
  cd -
}
bib() {
  [ "$1" = '-h' ] && echo -e "\x1b[33mbib\x1b[0m\n  Code formats the \x1b[1m\"references.bib\"\x1b[0m" && return
  cd "$src/paper"
  cat references.bib | bibtool -r .bibtoolrsc > references.bib.tmp
  mv references.bib.tmp references.bib
  cd -
}
cit() {
  [ "$1" = '-h' ] && echo -e "\x1b[33mcit\x1b[0m\n  Updates the citations of the Thesis Paper" && return
  #[ -e "$src/paper/build" ] && rm "$src/paper/build/"* -r
  mkdir -p "$src/paper/build"
  cd "$src/paper/build"
  cp ../*.{tex,bib} ../{pages,assets,lib}/* .
  lualatex -shell-escape thesis.tex
  biber thesis
  lualatex -shell-escape thesis.tex
  lualatex -shell-escape thesis.tex
  [ -e "$src/paper/build/thesis.pdf" ] && cp "$src/paper/build/thesis.pdf" ..
  mv *.svg ../assets
  rm *.tex *.ttf *.bib svg-inkscape -r
  cd -
}
pev() {
  local new=0
  [ ! -d "$src/py" ] && new=1

  [ "$new" -eq 1 ] && python3 -m venv "$src/py"
  source "$src/py/bin/activate"
  [ "$new" -eq 1 ] && pip install -r "$src/src/requirements.txt" --upgrade
}
uml() {
  [ -z "$1" ] && echo -e "\x1b[33muml\x1b[0m\n  Compiled PlantUML (.puml) files to svg" && return
  java -jar "$src/paper/assets/plantuml.jar" -tsvg "$1"
}
hos() {
  python3 -m http.server -d "$src/paper" 8000 &
}