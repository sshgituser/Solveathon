namespace Qolda;

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        ApplicationConfiguration.Initialize();
        var endpoint = LocalEndpoint.Default;
        string? config = null;
        var attachOnly = false;
        switch (args)
        {
            case []:
                break;
            case ["--config", var path] when !string.IsNullOrWhiteSpace(path) && !path.StartsWith("--", StringComparison.Ordinal):
                config = path;
                break;
            case ["--attach-port", var port] when LocalEndpoint.ForAttach(port) is { } allowed:
                endpoint = allowed;
                attachOnly = true;
                break;
            default:
                MessageBox.Show($"Использование: Qolda.exe [--config путь-к-JSON | --attach-port <порт>]\nРазрешённые порты: {LocalEndpoint.DefaultPort}, {LocalEndpoint.AlternatePort}.",
                    "Qolda", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 2;
        }

        using var singleInstance = new Mutex(false, endpoint.MutexName, out var first);
        if (!first) return 0;
        using var window = new MainWindow(endpoint, config, attachOnly);
        Application.Run(window);
        return 0;
    }
}
