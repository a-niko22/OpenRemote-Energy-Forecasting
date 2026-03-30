%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IDENTIFICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{styledarticle}[2025-12-23 A stylised article class]
\LoadClass{article}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRELIMINARY DECLARATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[dvipsnames]{xcolor} % For colors...
\RequirePackage[labelfont={sf,bf}]{caption} % Styled captions
\RequirePackage{amssymb} % For better emptyset
\RequirePackage{amsthm} % NEEDS TO BE LOADED BEFORE CLEVEREF
\RequirePackage{mathtools} % NEEDS TO BE LOADED BEFORE CLEVEREF
\RequirePackage{thmtools} % NEEDS TO BE LOADED BEFORE CLEVEREF
\RequirePackage[framemethod=tikz]{mdframed} % Needed for fancy definitions
\RequirePackage[T1]{fontenc} % Needed for certain fonts
\RequirePackage{changepage} % Needed for abstract
\RequirePackage[sf,bf]{titlesec} % Loads titlesec with sanserif family and bold face
\RequirePackage{graphicx} % Needed for colorbox
% Better margins (needs to be imported before the headers)
\RequirePackage[
    margin=3cm,
]{geometry}
\RequirePackage{pgfornament} % Needed for fancy line on title page
\RequirePackage[twoside]{fancyhdr} % For headers/footers
\RequirePackage{enumitem} % For list spacing
\RequirePackage{hyperref} % Should be loaded last, except for cleveref
\RequirePackage{cleveref} % NEEDS TO BE LOADED BEFORE `newtheorem` DEFINITIONS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print mode (changes things to use black and white and less ink)
\DeclareOption{print}{
    \colorlet{mylinkcolor}{black}
    \colorlet{mycitecolor}{black}
    \colorlet{myurlcolor}{black}
    \colorlet{boxlinecolor}{black}
    \colorlet{boxbg}{white}
    \colorlet{theoremcolor}{black}
    \newcommand*\boxlinewidth{2pt}
}
\DeclareOption*{%
    \PassOptionsToClass{\CurrentOption}{article}%
}
\ProcessOptions\relax

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MORE DECLARATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COLORS
% Colors for hyperlinks
\providecolor{mylinkcolor}{named}{violet}
\providecolor{mycitecolor}{named}{YellowOrange}
\providecolor{myurlcolor}{named}{Aquamarine}
\newcommand*\myshade{85}
% Hyperlink setup
\hypersetup{
  linkcolor  = mylinkcolor!\myshade!black,
  citecolor  = mycitecolor!\myshade!black,
  urlcolor   = myurlcolor!\myshade!black,
  colorlinks = true,
}
% Colors for theorems and definitions etc.
\providecolor{boxlinecolor}{named}{lightgray}
\colorlet{lightergray}{lightgray!20}
\providecolor{boxbg}{named}{lightergray}
\providecolor{theoremcolor}{named}{darkgray}
\providecommand*\boxlinewidth{3pt}

% TITLE
\makeatletter
\renewcommand*\@maketitle{%
  \newpage%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par%
  \vskip1em%
  \noindent%
  \pgfornament[width=\textwidth]{88}%
  \par%
  \vskip1.5em}
\makeatother
\let\oldmaketitle\maketitle
\renewcommand*\maketitle{%
    \oldmaketitle%
    \thispagestyle{empty}%
}

% HEADERS
\makeatletter
\def\title#1{\gdef\@title{#1}%
\providecommand\runningtitle{#1}%
}
\def\author#1{\gdef\@author{#1}
\providecommand\runningauthor{#1}%
}
\makeatother
\pagestyle{fancy}
\fancyhead[LE]{\runningauthor}
\fancyhead[LO]{\runningtitle}
\fancyhead[R]{\thepage}
\fancyfoot{}

% ABSTRACT
\renewenvironment{abstract}[1][4em]
{
\begin{adjustwidth}{#1}{#1}
\bfseries\sffamily
Abstract. \mdseries\rmfamily
}
{
\end{adjustwidth}
\par
}

% SECTIONS
\titleformat{\section}
{\sffamily\bfseries\Large}
{\colorbox{black}{\textcolor{white}\thesection}}
{1em}{}[\vspace{-.15em}\titlerule]

% ENVIRONMENTS
\newcommand*\proofsymbolstart{\blacktriangleright}
\newcommand*\proofsymbolend{\blacktriangleleft}
\mdfdefinestyle{shadedbox}{%
    frametitlefont=\bfseries\sffamily,
    theoremtitlefont=\mdseries,
    theoremseparator={},
    frametitlebelowskip=0pt,
    topline=false,  
    rightline=false,  
    bottomline=false,  
    leftline=true,  
    linecolor=boxlinecolor,  
    linewidth=\boxlinewidth,  
    backgroundcolor=boxbg%
}
\mdtheorem[style=shadedbox]{definition}{Definition}
\mdtheorem[style=shadedbox]{example}{Example}
\mdtheorem[style=shadedbox]{remark}{Remark}
\newtheoremstyle{customtheorem}% name of the style to be used
  {\topsep}% measure of space to leave above the theorem. E.g.: 3pt
  {\topsep}% measure of space to leave below the theorem. E.g.: 3pt
  {\itshape}% name of font to use in the body of the theorem
  {0pt}% measure of space to indent
  {\color{theoremcolor}\bfseries\sffamily}% name of head font
  {.}% punctuation between head and body
  { }% space after theorem head; " " = normal interword space
  {$\proofsymbolstart$ \thmname{#1}\thmnumber{ #2}\textmd{\thmnote{ #3}}}
\theoremstyle{customtheorem}
\newtheorem{theorem}{Theorem}[section]
\theoremstyle{customtheorem}
\newtheorem{proposition}[theorem]{Proposition}
\theoremstyle{customtheorem}
\newtheorem{lemma}[theorem]{Lemma}
\theoremstyle{customtheorem}
\newtheorem{claim}[theorem]{Claim}
\theoremstyle{customtheorem}
\newtheorem{conjecture}[theorem]{Conjecture}
\theoremstyle{customtheorem}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{customtheorem}
\newtheorem{invariant}{Invariant}
\Crefname{invariant}{Invariant}{Invariants}

% These are changes to the proof environment
\let\oldproofname=\proofname
% Changes the formatting for the 'Proof.' text at the start of the environment
\renewcommand{\proofname}{\color{theoremcolor}\sffamily\bfseries\upshape{\oldproofname}} 
% Changes the qed symbol to be a left pointing triangle
\renewcommand{\qedsymbol}{\color{theoremcolor}$\proofsymbolend$}

% CLEVEREF
\creflabelformat{equation}{#2\textup{#1}#3} % Removes brackets when referencing equations
\newcommand{\crefrangeconjunction}{--} % Use en-dash when referencing multiple items in a range

% MISC
% Make emptyset symbol nicer (you can still access it via \oldemptyset)
\let\oldemptyset\emptyset
\let\emptyset\varnothing

% smaller spacing between list items
\setlist{nosep}
% description font style
\setlist[description]{font=\color{theoremcolor}\bfseries\normalsize\sffamily}
% itemize bullet points
\renewcommand\labelitemi{\tiny$\blacksquare$}

% Proofsubparagraph definition
% Commands using @ need to go between \makeatletter and \makeatother
\makeatletter
\newcommand{\proofsubparagraph}{\@startsection{subparagraph}{5}{0pt}{3.25ex plus 1ex minus .2ex}{-1em}{\color{theoremcolor}\bfseries\normalsize\sffamily}}
\makeatother

% PACKAGE COMPATIBILITIES
\PassOptionsToPackage{noend}{algpseudocode}
\makeatletter
\def\ProcessPackages{
% algpseudocode
\IfPackageLoadedT{algpseudocode}{
% Changes commands to use new font
\newcommand*\keywordfont{\sffamily\bfseries}
\algrenewcommand\algorithmicend{{\keywordfont End}}
\algrenewcommand\algorithmicdo{{\keywordfont Do}}
\algrenewcommand\algorithmicwhile{{\keywordfont While}}
\algrenewcommand\algorithmicfor{{\keywordfont For}}
\algrenewcommand\algorithmicforall{{\keywordfont For all}}
\algrenewcommand\algorithmicloop{{\keywordfont Loop}}
\algrenewcommand\algorithmicrepeat{{\keywordfont Do}}
\algrenewcommand\algorithmicuntil{{\keywordfont While}}
\algrenewcommand\algorithmicprocedure{{\keywordfont Procedure}}
\algrenewcommand\algorithmicfunction{{\keywordfont Function}}
\algrenewcommand\algorithmicif{{\keywordfont If}}
\algrenewcommand\algorithmicthen{{\keywordfont Then}}
\algrenewcommand\algorithmicelse{{\keywordfont Else}}
\algrenewcommand\algorithmicrequire{{\keywordfont Precondition:}}
\algrenewcommand\algorithmicensure{{\keywordfont Postcondition:}}
\algrenewcommand\algorithmicreturn{{\keywordfont Return}}
\algrenewcommand\Return{\State\algorithmicreturn~}
\algrenewcommand\Call[2]{\textproc{##1}(##2)}
% Adding some additional commands
\newcommand*\Let[2]{\State ##1 $\coloneqq$ ##2}
\newcommand*\Ternary[4]{\Let{##1}{##2 \textbf{if} ##3 \textbf{else} ##4}}
\algnewcommand\Break{\State{\keywordfont Break}}
\algnewcommand{\LineComment}[1]{\Statex\hskip\ALG@thistlm \(\triangleright\) ##1}
% More cleveref
% Use plain to get reference to algorithm. 
% One star for algorithm and line number
% Two stars for just line number
\NewDocumentCommand{\Aref}{ s s m }{%
    \IfBooleanTF{##2}{}{%
        \Cref{##3}%
    }%
    \IfBooleanT{##1}{%
        \IfBooleanF{##2}{%
            , %
        }%
        Line~\ref{##3}%
    }%
}
}
\makeatother

% algorithm
\IfPackageLoadedT{algorithm}{
% Algorithm caption
\captionsetup[ruled]{labelfont={sf,bf}}
}
}