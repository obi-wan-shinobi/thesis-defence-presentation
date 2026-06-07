# deck.tex lives in slides/ and \usepackage{style}s slides/style.sty.
# Add slides/ to the TeX search path so `latexmk slides/deck.tex` from the
# repo root can find it (trailing colon keeps the default search path too).
$ENV{'TEXINPUTS'} = './slides:' . ($ENV{'TEXINPUTS'} // '') . ':';
