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
        alignItems: "center",
        marginTop: "20px",
    },
    textInput: {
        width: "100%",
        padding: "10px",
        margin: "10px 0",
        border: "1px solid #ccc",
        borderRadius: "4px",
        boxSizing: "border-box",
    },
    fileInput: {
        display: "none",
    },
    uploadButton: {
        backgroundColor: "#2196f3",
        color: "#fff",
        padding: "10px 15px",
        fontSize: "16px",
        border: "none",
        borderRadius: "4px",
        cursor: "pointer",
        marginBottom: "10px",
        transition: "background-color 0.3s",
        "&:hover": {
            backgroundColor: "#1565c0",
        },
    },
    predictButton: {
        backgroundColor: "#4caf50",
        color: "#fff",
        padding: "10px 15px",
        fontSize: "25px",
        fontFamily: "'Comic Sans MS', cursive",
        border: "none",
        borderRadius: "4px",
        cursor: "pointer",
        transition: "background-color 0.3s",
        "&:hover": {
            backgroundColor: "#45a049",
        },
    },
    preview: {
        maxHeight: "150px",
        maxWidth: "150px",
        alignItems: "center",
        marginTop: "20px",
    },
    help: {
        color: "#302f2f",
    },
    true: {
        color: "#31a354",
    },
    false: {
        color: "#de2d26",
    },
});

export default styles;
