package net.qolda.offline;

import android.os.CancellationSignal;
import android.os.OperationCanceledException;
import java.io.Closeable;
import java.io.IOException;

final class Cancel {
    final CancellationSignal signal = new CancellationSignal();
    private Closeable input;

    void check() { signal.throwIfCanceled(); }
    boolean cancelled() { return signal.isCanceled(); }

    synchronized void watch(Closeable stream) throws IOException {
        if (cancelled()) {
            stream.close();
            throw new OperationCanceledException();
        }
        input = stream;
    }

    synchronized void unwatch() { input = null; }

    void cancel() {
        signal.cancel();
        synchronized (this) {
            if (input != null) {
                try { input.close(); } catch (IOException ignored) { }
            }
        }
    }
}
