namespace Qolda;

internal sealed class LocalEndpoint
{
    public const string DefaultPort = "8765";
    public const string AlternatePort = "8870";
    private const string LegacyMutex = @"Local\Qolda.Desktop.T009";

    private LocalEndpoint(string port) => Port = port;

    public static LocalEndpoint Default { get; } = new(DefaultPort);

    public static LocalEndpoint? ForAttach(string port) => port switch
    {
        DefaultPort => Default,
        AlternatePort => new(AlternatePort),
        _ => null
    };

    private string Port { get; }
    public string Address => "127.0.0.1:" + Port;
    public string Origin => "http://" + Address;

    // Keep the legacy lock and profile on 8765 so old and new clients do not
    // create two windows for that port or lose the existing local preferences.
    public string MutexName => Port == DefaultPort ? LegacyMutex : LegacyMutex + "." + Port;
    public string ProfileName => Port == DefaultPort ? "WebView2" : "WebView2-" + Port;

    public bool Owns(string? url) => url == Origin ||
        (url is not null && url.StartsWith(Origin + "/", StringComparison.Ordinal));
}