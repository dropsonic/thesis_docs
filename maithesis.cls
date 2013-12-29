\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{maithesis}[2013/12/23 MAI Thesis]

% extreport options
\DeclareOption{draft}{
	\PassOptionsToClass{\CurrentOption}{extreport}
}
\DeclareOption{final}{
	\PassOptionsToClass{\CurrentOption}{extreport}
}
\DeclareOption{12pt}{
	\PassOptionsToClass{\CurrentOption}{extreport}
}
\DeclareOption{14pt}{
	\PassOptionsToClass{\CurrentOption}{extreport}
}

%% Fallback
\DeclareOption*{
  \ClassWarning{maithesis}{Unknown option '\CurrentOption'}
}

%% Execute default options
\ExecuteOptions{final,14pt}

%% Process given options
\ProcessOptions\relax

\LoadClass[oneside]{extreport}

% Основная часть **************************************************

\RequirePackage{pdfpages} % поддержка вставки страниц из pdf-файлов
\RequirePackage[onehalfspacing]{setspace} % 1,5 интервал
\RequirePackage[a4paper, top=2.0cm,bottom=2.0cm,left=2.0cm,right=1.0cm]{geometry} % поля
\RequirePackage{pdflscape} % поддержка альбомной ориентации страниц
\RequirePackage{indentfirst} % красная строка
\setlength\parindent{1.5cm} % установка величины отступа красной строки

\RequirePackage{etoolbox} % позволяет добавлять произвольный код в начало любой команды, и содержит прочие подобные доп. функции для программирования
\RequirePackage{suffix} % позволяет легко определять команды со звёздочкой (starred version of command)
\RequirePackage{afterpage} % float overload fix
\RequirePackage{placeins} % float barrier
\RequirePackage{caption} % подписи
\RequirePackage{subcaption} % подфлоаты во флоатах
\DeclareCaptionLabelSeparator{emdash}{ --- }
\RequirePackage{csvsimple} % таблицы из csv-файлов
%\RequirePackage[pages=some]{background}

% Команды для сохранения и вывода информации о работе (автор, руководитель и т.п.)
\def\thesistitle#1{\gdef\@thesistitle{#1}}
\def\@thesistitle{\@latex@error{No \noexpand\thesistitle given}\@ehc}

\def\superviser#1{\gdef\@superviser{#1}}
\def\@superviser{\@latex@error{No \noexpand\superviser given}\@ehc}

\def\specadviser#1{\gdef\@specadviser{#1}}
\def\@specadviser{\@latex@error{No \noexpand\specadviser given}\@ehc}

\def\specadviser#1{\gdef\@specadviser{#1}}
\def\@specadviser{\@latex@error{No \noexpand\specadviser given}\@ehc}

\def\headofsubfaculty#1{\gdef\@headofsubfaculty{#1}}
\def\@headofsubfaculty{\@latex@error{No \noexpand\headofsubfaculty given}\@ehc}

\def\universityname#1{\gdef\@universityname{#1}}
\def\@universityname{МАИ}

\def\specialtynumber#1{\gdef\@specialtynumber{#1}}
\def\@specialtynumber{\@latex@error{No \noexpand\specialtynumber given}\@ehc}

\def\thesisyear#1{\gdef\@thesisyear{#1}}
\def\@thesisyear{\@latex@error{No \noexpand\thesisyear given}\@ehc}

\def\groupnumber#1{\gdef\@groupnumber{#1}}
\def\@groupnumber{\@latex@error{No \noexpand\groupnumber given}\@ehc}

\def\keywords#1{\gdef\@keywords{#1}}
\def\@keywords{\@latex@error{No \noexpand\keywords given}\@ehc}

\newcommand{\thethesistitle}{\@thesistitle}
\newcommand{\thetitle}{\@title}
\newcommand{\theauthor}{\@author}
\newcommand{\thesuperviser}{\@superviser}
\newcommand{\thespecadviser}{\@specadviser}
\newcommand{\theheadofsubfaculty}{\@headofsubfaculty}
\newcommand{\theuniversityname}{\@universityname}
\newcommand{\thespecialtynumber}{\@specialtynumber}
\newcommand{\thethesisyear}{\@thesisyear}
\newcommand{\thegroupnumber}{\@groupnumber}
\newcommand{\thekeywords}{\@keywords}
%-----------------------------------------------------------------

% Шрифты ---------------------------------------------------------
\RequirePackage[cm-default]{fontspec}
\RequirePackage{xunicode}
\RequirePackage{xltxtra}
%\setromanfont[Mapping=tex-text]{Times New Roman}
\setmainfont[Mapping=tex-text]{Times New Roman}
\setsansfont[Mapping=tex-text]{Calibri}
\newfontfamily{\cyrillicfonttt}{Courier New}
\setmonofont[Mapping=tex-text]{Courier New}
\newfontfamily\GostTypeAFont{Arial} % GOST type A
\newfontfamily\GostTypeBFont{Arial} % GOST type B
\defaultfontfeatures{Scale=MatchLowercase, Mapping=tex-text} % одинаковый рост строчных букв у разных гарнитур, маппинги TeXовских лигатур вроде -- и ---
\RequirePackage{ulem} % поддержка подчёркиваний
%-----------------------------------------------------------------

% Поддержка русского языка и русскоязычных стилей ----------------
\RequirePackage{polyglossia}
\setmainlanguage[babelshorthands=true]{russian} % основной язык - русский
\setotherlanguage[variant=us]{english} % дополнительный язык - английский
%-----------------------------------------------------------------

% Формулы --------------------------------------------------------
\RequirePackage{amsmath}
\RequirePackage{amstext} % поддержка текста внутри формул
\RequirePackage{amssymb} % дополнительные символы в математических формулах
%\RequirePackage{icomma} % запятая в качестве десятичного разделителя
\RequirePackage{chngcntr} % управление нумерацией
\apptocmd{\subequations}{\renewcommand{\theequation}{\theparentequation\asbuk{equation}}}{}{} % нумерация подформул кириллицей (например, 5а, 5б, 5в и т.д.)
\counterwithout{equation}{chapter} % сквозная нумерация формул
% поддержка кириллицы в формулах (не работает, хотя должно)
%\RequirePackage{unicode-math}
%\setmathfont[math-style=TeX]{Cambria Math}
%\RequirePackage[warn]{mathtext}
%-----------------------------------------------------------------

% Формат заголовков ----------------------------------------------
\RequirePackage{titlesec}
\setcounter{secnumdepth}{3} % включает нумерацию subsubsection
\newcommand{\nhspacesize}{10pt}
\newcommand{\nohyphenation}{\righthyphenmin62}
\titleformat{\chapter}[hang]{\nohyphenation\sloppy\Large\bfseries}{\thechapter}{\nhspacesize}{} % nohyphenation - не переносить слова в названиях глав
\titleformat{\section}[hang]{\nohyphenation\sloppy\large\bfseries}{\thesection}{\nhspacesize}{}
\titleformat{\subsection}[hang]{\sloppy\normalsize\bfseries}{\thesubsection}{\nhspacesize}{}
\titleformat{\subsubsection}[hang]{\sloppy\normalsize\bfseries}{\thesubsubsection}{\nhspacesize}{}
\titlespacing*{\chapter}{\parindent}{-30pt}{*4}
\titlespacing*{\section}{\parindent}{*4}{*2}
\titlespacing*{\subsection}{\parindent}{*2}{*1}
\titlespacing*{\subsubsection}{\parindent}{*1}{*1}
%\setlength{\parskip}{1ex plus 0.5ex minus 0.2ex} % если вдруг нужен интервал между абзацами
%-----------------------------------------------------------------

% Новый формат заголовков - специальные разделы (реферат, вступление, заключение и т.п.)
\newcommand{\spchapterStar}[1]{
	\clearpage
	\begin{center}
		\nohyphenation{\sloppy{\textbf{\Large{#1}}}}
	\end{center}
	\par}
\newcommand{\spchapter}[1]{
	\spchapterStar{#1}
	\addcontentsline{toc}{chapter}{#1}}
\WithSuffix\newcommand\spchapter*[1]{\spchapterStar{#1}}
%-----------------------------------------------------------------

% Формат списков -------------------------------------------------
\RequirePackage{enumitem}
\setlist{nolistsep} % убрать лишний интервал между элементами списка
\setlist[itemize,1]{label={--}, labelindent=\parindent, leftmargin=*} % маркированные списки: символ - короткое тире, выравнивание символа по красной строке
\AddEnumerateCounter{\asbuk}{\asbuk}{д} % последний параметр - самый широкий символ в перечислении
\setlist[enumerate,1]{label={\asbuk*}), labelindent=\parindent, leftmargin=*}
\setlist[enumerate,2]{label={\arabic*}), leftmargin=\parindent}
\setlist[description]{labelindent=\parindent, leftmargin=0pt, font=\textmd}
\newenvironment{abbreviation}{%
\setlist[description]{labelindent=\parindent, leftmargin=0pt, font=\textbf}
\begin{description}}{\end{description}}
%-----------------------------------------------------------------

% Поддержка изображений ------------------------------------------
\RequirePackage{graphicx}
\graphicspath{{./images/}{./special/images/}{./economics/images/}{./labourprotection/images/}{./appendices/images/}} % пути к каталогам с изображениями
\RequirePackage{svg} % поддержка вставки векторных (SVG) изображений (из Inkscape) - команда includesvg
%-----------------------------------------------------------------

% Таблицы --------------------------------------------------------
%\RequirePackage{makecell}
%\RequirePackage{tabularx}
\RequirePackage{array} % дополнительные виды столбцов
\RequirePackage{longtable} % многостраничные таблицы
\RequirePackage{multirow} % объединение ячеек по вертикали
\RequirePackage{slashbox} % возможность разделить ячейку диагональной линией

%\newcolumntype{CL}[1]{>{\arraybackslash}m{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}m{#1}}
%\newcolumntype{CR}[1]{>{\raggedright\arraybackslash}m{#1}}
%\newcolumntype{TL}[1]{>{\arraybackslash}p{#1}}
%\newcolumntype{TC}[1]{>{\centering\arraybackslash}p{#1}
%\newcolumntype{TR}[1]{>{\raggedright\arraybackslash}p{#1}}
%\newcolumntype{BL}[1]{>{\arraybackslash}b{#1}}
%\newcolumntype{BC}[1]{>{\centering\arraybackslash}b{#1}}
%\newcolumntype{BR}[1]{>{\raggedright\arraybackslash}b{#1}}
%\newcommand{\ptw}[1]{#1\linewidth}

\setlength{\extrarowheight}{4pt}
%\renewcommand{\arraystretch}{1.5}
%\newcommand{\tn}{\tabularnewline}
%\newcommand{\tnhl}{\tabularnewline\hline}
%\newcommand{\tncl}[1]{\tabularnewline\cline{#1}}
%-----------------------------------------------------------------

% Формат рисунков и таблиц ---------------------------------------
\captionsetup{labelsep=emdash, textformat=simple, figurename=Рисунок, tablename=Таблица, figurewithin=none, tablewithin=none} % разделитель подписи и названия - короткое тире, заголовок - название float'a и номер, имена рисунков и таблиц по ГОСТу, сквозная нумерация рисунков и таблиц
\captionsetup[figure]{position=above}
\captionsetup[table]{singlelinecheck=false, position=top, justification=raggedright}
\renewcommand{\thesubfigure}{\asbuk{subfigure}} % использовать русский алфавит в качестве нумерации подрисунков
\captionsetup[subfigure]{labelformat=brace, font={small}}
% подписи многостраничных таблиц
\newcommand{\longtablecontcaption}{\captionsetup{justification=raggedleft, singlelinecheck=false,position=top,font=it}\caption*{Продолжение таблицы~\thetable}}
\newcommand{\longtableendcaption}{\captionsetup{justification=raggedleft, singlelinecheck=false,position=top,font=it}\caption*{Окончание таблицы~\thetable}}
\newcommand{\longtablehead}[2]{#1 \\ #2 \endfirsthead \longtableendcaption \\ #2 \endlasthead \longtablecontcaption \\ #2 \endhead}
%\DeclareCaptionLabelFormat{continued}{#1~#2 (\textit{продолжение})}
%\captionsetup[ContinuedFloat]{labelformat=continued}
\RequirePackage{floatrow} % пакет для настройки размещения float'ов и их подписей
\floatsetup[table]{style=plaintop, justification=justified} % название над таблицей, таблица выровнена по ширине влево
%-----------------------------------------------------------------

% Библиография и библиографические ссылки ------------------------
\RequirePackage{cite}
% Замена формата нумерации списка литературы с "[1]" на "1."
\renewcommand{\@biblabel}[1]{#1.}
\bibliographystyle{utf8gost705u}
\gappto\captionsrussian{\renewcommand{\bibname}{Список использованных источников}}
%-----------------------------------------------------------------

% Оглавление -----------------------------------------------------
\gappto\captionsrussian{\renewcommand{\contentsname}{Содержание}}
\RequirePackage[subfigure,titles]{tocloft}
\renewcommand{\cftchapleader}{\bfseries\cftdotfill{\cftdotsep}} % точечки не только у глав, но и у всего остального
% расстояние от номера до названия (чтобы номер на название на заезжал)
\setlength{\cftchapnumwidth}{2em}
\setlength{\cftsecnumwidth}{3em}
\setlength{\cftsubsecnumwidth}{4em}
%-----------------------------------------------------------------

% Подсчёт объектов (для реферата) --------------------------------
\RequirePackage{totcount}
%\regtotcounter{figure} % рисунки
%\regtotcounter{table} % таблицы
% подсчёт приложений
\newtotcounter{appendixcount}
\RequirePackage{apptools}
\pretocmd{\chapter}{\IfAppendix{\addtocounter{appendixcount}{1}}{}}{}{}
% подсчёт страниц
\RequirePackage{lastpage}
% подсчёт использованных источников
\newtotcounter{bibitemcount}
\pretocmd{\bibitem}{\addtocounter{bibitemcount}{1}}{}{}
% подсчёт рисунков и таблиц
\newtotcounter{figurecount}
\newtotcounter{tablecount}
\AfterEndEnvironment{figure}{\addtocounter{figurecount}{1}}
\AfterEndEnvironment{table}{\addtocounter{tablecount}{1}}
\AfterEndEnvironment{longtable}{\addtocounter{tablecount}{1}}
\newcommand{\pagestotal}{\pageref{LastPage}}
\newcommand{\figurestotal}{\total{figurecount}}
\newcommand{\tablestotal}{\total{tablecount}}
\newcommand{\bibitemstotal}{\total{bibitemcount}}
\newcommand{\appendicestotal}{\total{appendixcount}}
% составление реферата
\newcommand{\referat}[1][]%
{%
	\protect\spchapter*{\MakeUppercase{Реферат}}
	\sloppy
	{
		\theauthor\ \MakeUppercase{\thethesistitle}, дипломная работа: \pagestotal~с., \figurestotal~рис., \tablestotal~табл., \bibitemstotal~ист., \appendicestotal~прил.
		
		Ключевые слова: \MakeUppercase{\thekeywords}
	}
	\medskip
	
	#1
}
%-----------------------------------------------------------------

% Оформление приложений ------------------------------------------
\RequirePackage[title, titletoc]{appendix}
% задание своего формата заголовков для приложений
\apptocmd{\appendix}{
	\titleformat{\chapter}[display]{\nohyphenation\sloppy\normalsize\centering}{\MakeUppercase{\chaptertitlename} \thechapter}{\nhspacesize}{\bfseries}{}
	\counterwithin{equation}{chapter}
	\counterwithin{figure}{chapter}
	\counterwithin{table}{chapter}
	\counterwithin{algorithm}{chapter}
}{}{}
\apptocmd{\appendices}{
	\renewcommand{\thechapter}{\Asbuk{chapter}}
	\titlespacing*{\chapter}{\parindent}{-30pt}{*1}
}{}{}
%-----------------------------------------------------------------

% Псевдокод алгоритмов -------------------------------------------
\RequirePackage{algorithm}
\RequirePackage{algorithmic}
\BeforeBeginEnvironment{algorithm}{\medskip} % оставлять место перед листингом с псевдокодом
% перевод терминов псевдокода на русский
\renewcommand{\algorithmicrequire}{\textbf{Входные данные:}}
\renewcommand{\algorithmicensure}{\textbf{Выходные данные:}}
\renewcommand{\algorithmicend}{\textbf{конец}}
\renewcommand{\algorithmicif}{\textbf{если}}
\renewcommand{\algorithmicthen}{\textbf{то}}
\renewcommand{\algorithmicelse}{\textbf{иначе}}
\renewcommand{\algorithmicelsif}{\algorithmicelse\ \algorithmicif}
\renewcommand{\algorithmicendif}{\algorithmicend}
\renewcommand{\algorithmicfor}{\textbf{для}}
\renewcommand{\algorithmicforall}{\textbf{для всех}}
\renewcommand{\algorithmicdo}{\textbf{выполнять}}
\renewcommand{\algorithmicendfor}{\textbf{конец цикла}}
\renewcommand{\algorithmicwhile}{\textbf{до тех пор, пока}}
\renewcommand{\algorithmicendwhile}{\textbf{конец цикла}}
\renewcommand{\algorithmicloop}{\textbf{цикл}}
\renewcommand{\algorithmicendloop}{\textbf{конец цикла}}
\renewcommand{\algorithmicrepeat}{\textbf{повторять}}
\renewcommand{\algorithmicuntil}{\textbf{пока}}
\renewcommand{\algorithmicprint}{\textbf{вывести}}
\renewcommand{\algorithmicreturn}{\textbf{вернуть как результат}}
\renewcommand{\algorithmictrue}{\textbf{истина}}
\renewcommand{\algorithmicfalse}{\textbf{ложь}}
% конец перевода терминов псевдокода
\captionsetup[algorithm]{name = Листинг, labelsep = emdash}
\algsetup{indent=0.5\parindent}
%-----------------------------------------------------------------

% Листинги с исходным кодом --------------------------------------
\RequirePackage{listings}
\lstloadlanguages{[Sharp]C}
\lstset{language=[Sharp]C,
   morekeywords={yield,var,get,set,from,select,partial},
   breaklines=true,
   basicstyle=\footnotesize\ttfamily}
%-----------------------------------------------------------------

% Основные надписи -----------------------------------------------
\newcommand{\eskdheight}{287}
\newcommand{\eskdwidth}{200}

\newcommand{\eskdthin}{0.2}
\newcommand{\eskdhalfthin}{0.1}
\newcommand{\eskdthick}{0.8}
\newcommand{\eskdhalfthick}{0.4}

%\newcommand{\GostA}[1]{\GostTypeAFont\fontsize{20pt}{10pt}\fontshape{it}\selectfont\rotatebox{90}{#1}}
%\newcommand{\GostAsmall}[1]{\GostTypeAFont\fontsize{10pt}{10pt}\fontshape{it}\selectfont\rotatebox{90}{#1}}
%\newcommand{\GostB}[1]{\GostTypeBFont\fontsize{9pt}{10pt}\fontshape{it}\selectfont\rotatebox{90}{#1}}
%\newcommand{\GostBsmall}[1]{\GostTypeBFont\fontsize{8pt}{10pt}\fontshape{it}\selectfont\rotatebox{90}{#1}}
\newcommand{\GostA}[1]{\GostTypeAFont\fontsize{20pt}{10pt}\fontshape{it}\selectfont\rotatebox{90}{#1}}
\newcommand{\GostAsmall}[1]{\GostTypeAFont\fontsize{10pt}{10pt}\fontshape{it}\selectfont\rotatebox{90}{#1}}
\newcommand{\GostB}[1]{\GostTypeBFont\fontsize{9pt}{10pt}\fontshape{it}\selectfont\rotatebox{90}{#1}}
\newcommand{\GostBsmall}[1]{\GostTypeBFont\fontsize{8pt}{10pt}\fontshape{it}\selectfont\rotatebox{90}{#1}}

\newcommand{\eskd@stampbox}{%
	\unitlength 1mm
	\begin{picture}(0,0)(-\eskdwidth,0)
		\linethickness{\eskdthick mm}
		% Рамка
		\put(0,0){\line(-1,0){\eskdwidth}}
		\put(0,-\eskdheight){\line(-1,0){\eskdwidth}}
		\put(0,0){\line(0,-1){\eskdheight}}
		\put(-\eskdwidth,0){\line(0,-1){\eskdheight}}
		% Углы рамки
		\put(\eskdhalfthick,0){\line(-1,0){\eskdhalfthick}}
		\put(\eskdhalfthick,-\eskdheight){\line(-1,0){\eskdhalfthick}}
		\put(-\eskdwidth,0){\line(-1,0){\eskdhalfthick}}
		\put(-\eskdwidth,-\eskdheight){\line(-1,0){\eskdhalfthick}}
		
		% Таблица
		% толстые линии - длинные горизонтальные
		\put(-25,0){\line(0,-1){185}}
		\put(-40,0){\line(0,-1){185}}
		% толстые линии - короткие горизонтальные
		\put(-15,0){\line(0,-1){50}}
		\put(-20,0){\line(0,-1){50}}
		\put(-30,-120){\line(0,-1){65}}
		% тонкие линии - горизонтальные
		\linethickness{\eskdthin mm}
		\multiput(-5,-120)(-5,0){7}{\line(0,-1){65}}
		\linethickness{\eskdthick mm}
		% толстые линии - вертикальные
		\put(0,-185){\line(-1,0){40}}
		\put(-40,-178){\line(1,0){15}}
		\put(0,-168){\line(-1,0){40}}
		\put(0,-145){\line(-1,0){40}}
		\put(0,-130){\line(-1,0){40}}
		\put(0,-120){\line(-1,0){40}}
		\put(0,-50){\line(-1,0){25}}
		\put(-15,-35){\line(-1,0){10}}
		\put(-15,-20){\line(-1,0){10}}
		% тонкие линии - вертикальные
		\linethickness{\eskdthin mm}
		\put(-15,-45){\line(-1,0){5}}
		\put(-15,-40){\line(-1,0){5}}
		\linethickness{\eskdthick mm}
		% Углы таблицы
		\put(-40,-185){\line(-1,0){\eskdhalfthick}}
		
		% Надписи
		% левые мелкие
		\put(-28.5,-184.5){\GostBsmall{Изм.}}
		\put(-28.6,-177){\GostB{Лист}}
		\put(-28.8,-166.5){\GostB{№ докум.}}
		\put(-28.8,-143.5){\GostB{Подп.}}
		\put(-28.6,-129){\GostB{Дата}}
		\put(-23.6,-184){\GostB{Разраб.}}
		\put(-18.6,-184){\GostB{Руковод.}}
		\put(-13.6,-184){\GostB{Консульт.}}
		\put(-3.6,-184){\GostB{Утв.}}
		% правые мелкие
		\put(-23.6,-48.5){\GostB{Лит.}}
		\put(-23.6,-33.5){\GostB{Лист}}
		\put(-23.6,-15.8){\GostB{Листов}}
		% динамические поля
		\put(-23.4,-167){\GostBsmall{\theauthor}}
		\put(-18.4,-167){\GostBsmall{\thesuperviser}}
		\put(-13.4,-167){\GostBsmall{\thespecadviser}}
		\put(-3.4,-167){\GostBsmall{\theheadofsubfaculty}}
		\put(-35,-117){\GostA{\parbox{114mm}{\centering\theuniversityname-\thespecialtynumber-\thethesisyear}}}
		\put(-23,-118){\GostAsmall{\parbox{65mm}{\thethesistitle}}}
		\put(-10,-48){\GostA{\parbox{46mm}{\centering\thegroupnumber}}}
		\put(-18.4,-34){\GostB{\parbox{12.7mm}{\centering\thepage}}}
		\put(-18.4,-19){\GostB{\parbox{17mm}{\centering\pagestotal}}}
	\end{picture}
}

\newcommand{\includepresentation}[1]{%
	\newgeometry{top=0cm,bottom=0.6cm,left=0.6cm-\parindent,right=0cm}
	\includepdf[pages=-, pagecommand={%
		\thispagestyle{empty}
		\begin{picture}(0,0)
			\put(0,0){\eskd@stampbox}
		\end{picture}
	},landscape,scale=0.7, offset=-1.7cm -0.5cm]{#1}
	\restoregeometry
}
%-----------------------------------------------------------------

\endinput