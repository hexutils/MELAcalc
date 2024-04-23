#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "JHUGenLexiconOptionParser.h"
#include "JHUGenLexiconTranslator.h"

namespace py = pybind11;
using namespace std;



PYBIND11_MODULE(Lexicon, m){
    py::class_<JHUGenLexiconOptionParser>(m, "LexiconParser")
        .def(py::init)
}