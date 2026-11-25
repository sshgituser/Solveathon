using System.Runtime.InteropServices;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace Qolda;

internal sealed class MainWindow : Form
{
    private readonly WebView2 web = new() { Dock = DockStyle.Fill, Visible = false };
    private readonly Label message = new() { Dock = DockStyle.Fill, Padding = new Padding(12, 8, 8, 8), TextAlign = ContentAlignment.MiddleLeft };
    private readonly Button retry = new() { Text = "Подключиться", Dock = DockStyle.Right, Width = 152, Visible = false };
    private readonly BackendConnection backend;
    private readonly CancellationTokenSource lifetime = new();
    private readonly System.Windows.Forms.Timer heartbeat = new() { Interval = 5000 };
    private readonly string? config;
    private readonly LocalEndpoint endpoint;
    private readonly bool attachOnly;
    private bool busy;
    private bool frontendLoaded;

    public MainWindow(LocalEndpoint endpoint, string? config, bool attachOnly)
    {
        this.endpoint = endpoint;
        this.config = config;
        this.attachOnly = attachOnly;
        backend = new BackendConnection(endpoint);
        Text = "Qolda";
        AccessibleName = "Qolda Desktop";
        StartPosition = FormStartPosition.CenterScreen;
        ClientSize = new Size(1200, 820);
        MinimumSize = new Size(650, 520);
        BackColor = Color.FromArgb(12, 42, 51);
        var status = new Panel { Dock = DockStyle.Bottom, Height = 68, Padding = new Padding(0, 3, 8, 3) };
        status.Controls.Add(message);
        status.Controls.Add(retry);
        Controls.Add(web);
        Controls.Add(status);
        Shown += async (_, _) => await Connect(allowStart: true);
        retry.Click += async (_, _) => await Connect(allowStart: false);
        heartbeat.Tick += async (_, _) => await RefreshHealth();
        ShowStatus("подключение", "Проверяем локальный движок Qolda…");
    }

    private async Task Connect(bool allowStart)
    {
        if (busy || lifetime.IsCancellationRequested) return;
        busy = true;
        heartbeat.Stop();
        retry.Visible = false;
        var cancellation = lifetime.Token;
        try
        {
            ShowStatus("подключение", "Проверяем локальный движок Qolda…");
            var engine = await backend.Probe(cancellation);
            if (engine == EngineState.Refused && allowStart && !attachOnly)
            {
                var (options, baseDir) = BackendOptions.Load(config);
                if (!options.AttachOnly)
                {
                    ShowStatus("запуск движка", "Загружаем локальный CUDA-движок и модель A. Это может занять до 90 секунд.");
                    using var process = options.Start(baseDir);
                    engine = await backend.WaitForStartup(process, cancellation);
                }
            }

            if (engine is not (EngineState.Ready or EngineState.NoModel))
                throw new StartupException(DescribeConnectionError(engine));
            if (!frontendLoaded)
            {
                if (!await backend.HasFrontend(cancellation))
                    throw new StartupException("Движок ответил, но не раздаёт интерфейс Qolda. Нужен обычный запуск backend, без --api-only.");
                ShowStatus("загрузка интерфейса", "Открываем интерфейс Qolda в настольном окне…");
                await InitializeWebView();
                await LoadFrontend(cancellation);
                frontendLoaded = true;
                web.Visible = true;
            }
            ShowEngineState(engine);
            heartbeat.Start();
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested) { }
        catch (ObjectDisposedException) when (cancellation.IsCancellationRequested) { }
        catch (StartupException e) when (!cancellation.IsCancellationRequested)
        {
            ShowStatus("движок недоступен", e.Message, Tone.Error, canRetry: true);
        }
        catch (Exception e) when (!cancellation.IsCancellationRequested &&
            (e is COMException or InvalidOperationException or IOException or UnauthorizedAccessException or DllNotFoundException))
        {
            ShowStatus("не удалось открыть интерфейс", "Проверьте Microsoft Edge WebView2 Runtime и доступ к локальному профилю Qolda.", Tone.Error, canRetry: true);
        }
        finally { busy = false; }
    }

    private async Task InitializeWebView()
    {
        if (web.CoreWebView2 is not null) return;
        try { _ = CoreWebView2Environment.GetAvailableBrowserVersionString(); }
        catch (WebView2RuntimeNotFoundException)
        {
            throw new StartupException("Не установлен Microsoft Edge WebView2 Runtime. Установите его из локального установщика и повторите подключение.");
        }
        var profileDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Qolda", endpoint.ProfileName);
        var options = new CoreWebView2EnvironmentOptions(
            "--disable-background-networking --disable-component-update --no-proxy-server");
        var environment = await CoreWebView2Environment.CreateAsync(userDataFolder: profileDir, options: options);
        lifetime.Token.ThrowIfCancellationRequested();
        if (!string.Equals(Path.TrimEndingDirectorySeparator(environment.UserDataFolder), profileDir, StringComparison.OrdinalIgnoreCase))
            throw new StartupException("WebView2 переопределил каталог профиля Qolda. Используйте отдельный профиль для этого порта.");
        _ = web.Handle;
        await web.EnsureCoreWebView2Async(environment);
        lifetime.Token.ThrowIfCancellationRequested();
        var core = web.CoreWebView2 ?? throw new StartupException("WebView2 не удалось инициализировать.");
        var settings = core.Settings;
        settings.AreHostObjectsAllowed = false;
        settings.IsWebMessageEnabled = false;
        settings.AreDevToolsEnabled = false;
        settings.AreDefaultContextMenusEnabled = false;
        settings.AreBrowserAcceleratorKeysEnabled = false;
        settings.IsBuiltInErrorPageEnabled = false;
        settings.IsGeneralAutofillEnabled = false;
        settings.IsPasswordAutosaveEnabled = false;
        core.NavigationStarting += (_, e) => e.Cancel = !endpoint.Owns(e.Uri);
        core.FrameNavigationStarting += (_, e) => e.Cancel = !endpoint.Owns(e.Uri);
        core.NewWindowRequested += (_, e) => e.Handled = true;
        core.LaunchingExternalUriScheme += (_, e) => e.Cancel = true;
        core.PermissionRequested += (_, e) =>
        {
            e.SavesInProfile = false;
            e.State = e.PermissionKind == CoreWebView2PermissionKind.Camera && e.IsUserInitiated &&
                      endpoint.Owns(e.Uri) && endpoint.Owns(core.Source)
                ? CoreWebView2PermissionState.Default // Let WebView2 ask; the host never grants capture.
                : CoreWebView2PermissionState.Deny;
        };
        core.AddWebResourceRequestedFilter("*", CoreWebView2WebResourceContext.All, CoreWebView2WebResourceRequestSourceKinds.All);
        core.WebResourceRequested += (_, e) =>
        {
            var url = e.Request.Uri;
            if (endpoint.Owns(url) || (url.StartsWith("blob:", StringComparison.Ordinal) && endpoint.Owns(url[5..])) ||
                url.StartsWith("data:", StringComparison.Ordinal)) return;
            e.Response = environment.CreateWebResourceResponse(null, 403, "Forbidden", "Cache-Control: no-store");
        };
    }

    private async Task LoadFrontend(CancellationToken cancellation)
    {
        var core = web.CoreWebView2;
        var finished = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        void OnNavigationCompleted(object? sender, CoreWebView2NavigationCompletedEventArgs e) => finished.TrySetResult(e.IsSuccess);
        core.NavigationCompleted += OnNavigationCompleted;
        try
        {
            core.Navigate(endpoint.Origin + "/");
            if (!await finished.Task.WaitAsync(TimeSpan.FromSeconds(20), cancellation) || !endpoint.Owns(core.Source))
                throw new StartupException("Не удалось загрузить интерфейс Qolda из локального движка.");
            await WaitForFrontendRender(core, cancellation);
        }
        catch (TimeoutException)
        {
            throw new StartupException("Интерфейс Qolda не загрузился вовремя. Проверьте папки frontend и assets движка.");
        }
        finally { core.NavigationCompleted -= OnNavigationCompleted; }
    }

    private async Task WaitForFrontendRender(CoreWebView2 core, CancellationToken cancellation)
    {
        // The JS module can mount after NavigationCompleted; the localized title is not a readiness marker.
        string marker = $"location.origin === '{endpoint.Origin}' && document.querySelector('meta[name=seriq-profile]')?.content === 'local' && !!document.querySelector('#view > *')";
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        deadline.CancelAfter(TimeSpan.FromSeconds(5));
        try
        {
            while (endpoint.Owns(core.Source))
            {
                if (await core.ExecuteScriptAsync(marker).WaitAsync(deadline.Token) == "true" && endpoint.Owns(core.Source))
                    return;
                await Task.Delay(100, deadline.Token);
            }
        }
        catch (OperationCanceledException) when (!cancellation.IsCancellationRequested && deadline.IsCancellationRequested) { }
        throw new StartupException("Движок раздаёт несовместимый или неполный интерфейс Qolda.");
    }

    private async Task RefreshHealth()
    {
        if (busy || !frontendLoaded || lifetime.IsCancellationRequested) return;
        busy = true;
        try { ShowEngineState(await backend.Probe(lifetime.Token)); }
        catch (OperationCanceledException) when (lifetime.IsCancellationRequested) { }
        catch (ObjectDisposedException) when (lifetime.IsCancellationRequested) { }
        finally { busy = false; }
    }

    private string DescribeConnectionError(EngineState engine) => engine switch
    {
        EngineState.WrongService => $"Служба на {endpoint.Address} не соответствует API Qolda seriq.v1. Этот процесс не изменён.",
        EngineState.Unreachable => $"Служба на {endpoint.Address} не ответила. Она не перезапущена; дождитесь запуска движка и нажмите «Подключиться».",
        _ when attachOnly => $"Нет движка на {endpoint.Address}. Этот режим не запускает движок. Когда он будет готов, нажмите «Подключиться».",
        _ => $"Нет движка на {endpoint.Address}. Запустите start-engine.cmd из engine-win64 и нажмите «Подключиться»."
    };

    private void ShowEngineState(EngineState engine)
    {
        if (engine == EngineState.Ready)
            ShowStatus("модель A готова", $"Локальный движок · {endpoint.Address} · модель A готова");
        else if (engine == EngineState.NoModel)
            ShowStatus("модель A недоступна", "Движок ответил, но модель A не готова. Проверьте bundle, веса, FFprobe и NVIDIA CUDA. Анализ недоступен.", Tone.Warning, canRetry: true);
        else
            ShowStatus("движок недоступен", DescribeConnectionError(engine), Tone.Error, canRetry: true);
    }

    private enum Tone { Normal, Warning, Error }

    private void ShowStatus(string caption, string text, Tone tone = Tone.Normal, bool canRetry = false)
    {
        if (lifetime.IsCancellationRequested) return;
        Text = $"Qolda — {endpoint.Address} — {caption}";
        message.Text = text;
        message.AccessibleName = text;
        message.BackColor = tone switch
        {
            Tone.Warning => Color.FromArgb(80, 53, 13),
            Tone.Error => Color.FromArgb(101, 37, 44),
            _ => Color.FromArgb(12, 42, 51)
        };
        message.ForeColor = Color.White;
        retry.Visible = canRetry;
    }

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        lifetime.Cancel();
        heartbeat.Dispose();
        backend.Dispose();
        // The engine is shared with the Web client; closing Qolda never stops it.
        base.OnFormClosed(e);
    }
}
