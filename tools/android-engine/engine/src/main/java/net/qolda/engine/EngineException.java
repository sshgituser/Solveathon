package net.qolda.engine;

/** Stable failure surface of the engine. {@link #code()} is a wire-level identifier. */
public final class EngineException extends Exception {
    public static final String MODEL_UNAVAILABLE = "model_unavailable";
    public static final String INVALID_ARGUMENT = "invalid_argument";
    public static final String CLOSED = "closed";

    private final String code;

    /** Used by the native layer for GPU failures; always {@link #MODEL_UNAVAILABLE}. */
    public EngineException(String message) {
        super(message);
        this.code = MODEL_UNAVAILABLE;
    }

    public EngineException(String code, String message) {
        super(message);
        this.code = code;
    }

    public EngineException(String code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public String code() {
        return code;
    }
}
