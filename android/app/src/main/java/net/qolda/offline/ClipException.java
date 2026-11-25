package net.qolda.offline;

final class ClipException extends Exception {
    final String code;
    ClipException(String code) { super(code); this.code = code; }
}
