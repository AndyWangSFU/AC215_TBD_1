const styles = theme => ({
    root: {
        flexGrow: 1,
        minHeight: "100vh"
    },
    grow: {
        flexGrow: 1,
    },
    main: {
        // Add any main styles if needed
    },
    container: {
        backgroundColor: "#ffffff",
        paddingTop: "30px",
        paddingBottom: "20px",
    },
    textInputContainer: {
        display: "flex",
        flexDirection: "column",
        alignItems: "left",

        // alignItems: "left",
        marginTop: "20px",
    },
    textInput: {
        width: "80%",
        padding: "10px",
        margin: "10px 0",
        marginLeft: "10%",
        border: "1px solid #ccc",
        borderRadius: "4px",
        boxSizing: "border-box",
        alignItems: "center",
        rows: "5",
        wrap: "hard",
    },
    fileInput: {
        display: "none",
    },
    uploadButton: {
        backgroundColor: "#2196f3",
        color: "#fff",
        padding: "10px 15px",
        fontSize: "16px",
        fontFamily: "'Roboto Slab', sans-serif",
        border: "none",
        borderRadius: "4px",
        cursor: "pointer",
        marginBottom: "10px",
        transition: "background-color 0.3s",
        "&:hover": {
            backgroundColor: "#1565c0",
        },
        alignItems: "center",
        maxWidth: "300px",
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.15)',
        width: "20%",
        marginLeft: "36.5%",
        marginRight: "35%",
    },
    predictButton: {
        backgroundColor: "#4caf50",
        color: "#fff",
        padding: "10px 15px",
        fontSize: "25px",
        fontFamily: "'Comic Sans MS', cursive",
        border: "none",
        borderRadius: "10px",
        cursor: "pointer",
        transition: "background-color 0.3s",
        "&:hover": {
            backgroundColor: "#45a049",
        },
        maxWidth: "300px",
        marginTop: "10px",
        marginBottom: "20px",
        alignItems: "center",
        width: "40%",
        marginLeft: "30%",
    },
    preview: {
        maxHeight: "200px",
        maxWidth: "50%",
        alignItems: "center",
        justifyContent: "center",
        marginLeft: "30%",
        marginRight: "30%",
        marginBottom: "40px",
        
    },
    help: {
        color: '#302f2f',
        fontFamily: "'Roboto Slab', sans-serif",
        fontSize: '20px',
        fontWeight: 'bold',
        textAlign: 'left',
        marginTop: '10px',
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.15)',
    },
    help1: {
        color: '#302f2f',
        fontFamily: "'Roboto Slab', sans-serif",
        fontSize: '16px',
        fontWeight: 'bold',
        textAlign: 'left',
        marginTop: '10px',
        padding: '20px',
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.15)',
    },
    true: {
        color: "#31a354",
    },
    false: {
        color: "#de2d26",
    },
});

export default styles;
