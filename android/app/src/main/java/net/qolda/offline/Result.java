package net.qolda.offline;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

final class Result {
    static final String API_VERSION = "seriq.v1";
    static final String MODEL_VERSION = "seriq-e02-forearm-1.0.0+8dd234d4a3e1";
    final String label, reason, warning;
    final Double confidence;

    private Result(String label, Double confidence, String reason, String warning) {
        this.label = label;
        this.confidence = confidence;
        this.reason = reason;
        this.warning = warning;
    }

    // backend/runtime.py::wire_result. Geometry refusals are not incomplete predictions.
    static Result refused(String reason) {
        switch (reason) {
            case "no_visible_forearm_pair": return new Result(null, null, null, "no_pose");
            case "insufficient_visible_forearm_pair":
            case "incomplete_forearm_coverage":
            case "ambiguous_forearm_tracks":
                return new Result("uncertain", null, "low_visibility", "low_visibility");
            default: return new Result("uncertain", null, "unscorable", null);
        }
    }

    static Result score(float logit) {
        // ml/api.py uses the float32 sigmoid before computing the normalized margin.
        float p = (float) (1.0 / (1.0 + Math.exp(-logit)));
        double confidence = Math.abs(p - .5) * 2;
        if (confidence < .10) return new Result("uncertain", confidence, "low_confidence", null);
        return new Result(p >= .5 ? "complete" : "incomplete", confidence, null, null);
    }

    JSONObject wire() throws JSONException {
        JSONArray attempts = new JSONArray(), warnings = new JSONArray();
        if (label != null) attempts.put(attempt());
        if (warning != null) warnings.put(warning);
        JSONObject counts = new JSONObject().put("total", attempts.length())
                .put("complete", "complete".equals(label) ? 1 : 0)
                .put("incomplete", "incomplete".equals(label) ? 1 : 0)
                .put("uncertain", "uncertain".equals(label) ? 1 : 0);
        return new JSONObject().put("version", API_VERSION).put("model", "scratch")
                .put("model_version", MODEL_VERSION).put("attempts", attempts)
                .put("counts", counts).put("warnings", warnings);
    }

    JSONObject attempt() throws JSONException {
        return new JSONObject().put("index", 1).put("label", label)
                .put("confidence", confidence == null ? JSONObject.NULL : confidence)
                .put("reason", reason == null ? JSONObject.NULL : reason);
    }

    static Result readAttempt(JSONObject item) throws JSONException {
        String label = item.getString("label");
        Double confidence = item.isNull("confidence") ? null : item.getDouble("confidence");
        String reason = item.isNull("reason") ? null : item.getString("reason");
        return new Result(label, confidence, reason, "low_visibility".equals(reason) ? reason : null);
    }

    static JSONObject wireError(String code) throws JSONException {
        return new JSONObject().put("version", API_VERSION).put("error", new JSONObject().put("code", code));
    }
}
