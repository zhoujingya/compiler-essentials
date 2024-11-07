#include "TinyTablegenEmitter.h"

namespace tinytblgen {

void Emitter::EmitTokensAndKeywordFilter(raw_ostream &OS,
                                         RecordKeeper &Records) {
  OS << "{\n  \"tokens\": [\n";

  bool first = true;
  for (const auto &Record : Records.getDefs()) {
    if (!first) {
      OS << ",\n";
    }
    first = false;

    OS << "    {\n";
    OS << "      \"name\": \"" << Record.second->getName() << "\"\n";
    // Add more fields as needed, for example:
    // OS << "      \"keyword\": \"" << Record->getField("keyword").getValue()
    // << "\"\n";
    OS << "    }";
  }

  OS << "\n  ]\n}\n";
}
} // namespace tinytblgen
