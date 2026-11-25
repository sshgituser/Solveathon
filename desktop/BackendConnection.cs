using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;

namespace Qolda;

internal enum EngineState { Refused, Unreachable, WrongService, NoModel, Ready }

internal sealed class BackendConnection(LocalEndpoint endpoint) : IDisposable
{
    private readonly HttpClient http = new(new SocketsHttpHandler
    {
        UseProxy = false,
        UseCookies = false,
        AllowAutoRedirect = false
    }) { BaseAddress = new Uri(endpoint.Origin), Timeout = Timeout.InfiniteTimeSpan };

    public async Task<EngineState> Probe(CancellationToken cancellation)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        deadline.CancelAfter(TimeSpan.FromSeconds(3));
        try
        {
            using var response = await http.GetAsync("/api/v1/health", HttpCompletionOption.ResponseHeadersRead, deadline.Token);
            if (response.StatusCode != HttpStatusCode.OK || response.Content.Headers.ContentType?.MediaType != "application/json")
                return EngineState.WrongService;
            await using var body = await response.Content.ReadAsStreamAsync(deadline.Token);
            var buffer = new byte[2049];
            var size = await body.ReadAtLeastAsync(buffer, buffer.Length, false, deadline.Token);
            if (size == buffer.Length) return EngineState.WrongService;
            using var json = JsonDocument.Parse(buffer.AsMemory(0, size));
            var root = json.RootElement;
            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("version", out var version) ||
                version.ValueKind != JsonValueKind.String || version.GetString() != "seriq.v1" ||
                !root.TryGetProperty("models", out var models) || models.ValueKind != JsonValueKind.Object ||
                !TryBoolean(models, "scratch", out var scratch) || !TryBoolean(models, "transfer", out _))
                return EngineState.WrongService;
            return scratch ? EngineState.Ready : EngineState.NoModel;
        }
        catch (JsonException) { return EngineState.WrongService; }
        catch (HttpRequestException e) when (e.InnerException is SocketException { SocketErrorCode: SocketError.ConnectionRefused })
        {
            return EngineState.Refused;
        }
        catch (Exception e) when (!cancellation.IsCancellationRequested &&
            (e is HttpRequestException or IOException or OperationCanceledException))
        {
            return EngineState.Unreachable;
        }
    }

    private static bool TryBoolean(JsonElement obj, string key, out bool value)
    {
        value = false;
        if (!obj.TryGetProperty(key, out var entry) || entry.ValueKind is not (JsonValueKind.True or JsonValueKind.False)) return false;
        value = entry.GetBoolean();
        return true;
    }

    public async Task<bool> HasFrontend(CancellationToken cancellation)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        deadline.CancelAfter(TimeSpan.FromSeconds(3));
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Head, "/");
            using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token);
            return response.StatusCode == HttpStatusCode.OK && response.Content.Headers.ContentType?.MediaType == "text/html";
        }
        catch (Exception e) when (!cancellation.IsCancellationRequested &&
            (e is HttpRequestException or IOException or OperationCanceledException)) { return false; }
    }

    public async Task<EngineState> WaitForStartup(Process process, CancellationToken cancellation)
    {
        var elapsed = Stopwatch.StartNew();
        while (elapsed.Elapsed < TimeSpan.FromSeconds(90))
        {
            var state = await Probe(cancellation);
            if (state is EngineState.Ready or EngineState.NoModel or EngineState.WrongService) return state;
            if (process.HasExited)
                throw new StartupException($"Движок завершился при запуске (код {process.ExitCode}). Проверьте Python, CUDA и журнал ручного запуска start-engine.cmd.");
            await Task.Delay(750, cancellation);
        }
        throw new StartupException("Движок не ответил за 90 секунд. Не перезапускайте его: дождитесь загрузки и нажмите «Подключиться».");
    }

    public void Dispose() => http.Dispose();
}
